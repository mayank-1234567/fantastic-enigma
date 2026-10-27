import io
import os
import random
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import timedelta

import chess
import chess.engine
import fitz  # PyMuPDF
import numpy as np
from flask import Flask, jsonify, render_template_string, request, session
from PIL import Image

# Original AndrewSpano/2d-chess-ocr repository is cloned to this path in Docker.
OCR_REPO = Path(os.environ.get("CHESS_OCR_REPO", "/opt/2d-chess-ocr"))
if str(OCR_REPO) not in os.sys.path:
    os.sys.path.insert(0, str(OCR_REPO))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-render-secret")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_PDF_MB", "25")) * 1024 * 1024

TARGET_PLIES = 10
ENGINE_TIME = float(os.environ.get("ENGINE_TIME", "0.35"))
ACCEPTABLE_LOSS_CP = int(os.environ.get("ACCEPTABLE_LOSS_CP", "125"))
MULTIPV = 8
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "160"))
PDF_DPI_SCALE = float(os.environ.get("PDF_DPI_SCALE", "1.65"))

ENGINE = None
ENGINE_LOCK = threading.Lock()
OCR = None
OCR_LOCK = threading.Lock()
JOBS = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-ocr")


def find_stockfish():
    candidates = [os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish"),
                  "/usr/games/stockfish", "/usr/bin/stockfish"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def get_engine():
    global ENGINE
    if ENGINE is not None:
        try:
            if ENGINE.id:
                return ENGINE
        except Exception:
            ENGINE = None
    path = find_stockfish()
    if not path:
        raise RuntimeError("Stockfish was not found. Check the Render Docker build and STOCKFISH_PATH.")
    ENGINE = chess.engine.SimpleEngine.popen_uci(path, timeout=30)
    opts = {}
    if "Threads" in ENGINE.options:
        opts["Threads"] = 1
    if "Hash" in ENGINE.options:
        opts["Hash"] = 96
    if opts:
        ENGINE.configure(opts)
    return ENGINE


def get_ocr():
    """Load the repository's own YOLO inference wrapper and its trained weights."""
    global OCR
    if OCR is not None:
        return OCR
    with OCR_LOCK:
        if OCR is not None:
            return OCR
        model_override = os.environ.get("CHESS_OCR_MODEL")
        possibilities = [Path(model_override)] if model_override else []
        possibilities += [
            OCR_REPO / "finetuned-models" / "yolo26m-finetuned.pt",
            OCR_REPO / "finetuned-models" / "yolo26m-finetuned.onnx",
        ]
        model_path = next((p for p in possibilities if p and p.is_file()), None)
        if model_path is None:
            raise RuntimeError(
                "The 2d-chess-ocr model weights were not found. The Docker build must run "
                "the repository's download_models.py and place weights in finetuned-models/."
            )
        try:
            from model_inference import Chess2DOCRInference
        except Exception as exc:
            raise RuntimeError(
                f"Could not import model_inference.py from {OCR_REPO}: {exc}"
            ) from exc
        OCR = Chess2DOCRInference(model_path, conf=0.25, device="cpu")
        return OCR


def detector_turn(det):
    turn = str(getattr(det, "turn", "w") or "w").lower()
    confidence = float(getattr(det, "turn_conf", 0.5) or 0.5)
    # Only trust a model turn when there is better than coin-flip confidence.
    if turn not in ("w", "b"):
        turn = "w"
    return turn, confidence


def process_pdf(pdf_bytes, job_id):
    try:
        detector = get_ocr()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("This PDF has no pages.")
        if doc.page_count > MAX_PDF_PAGES:
            raise ValueError(
                f"This PDF has {doc.page_count} pages. The current limit is {MAX_PDF_PAGES}; "
                "split the book into smaller PDFs and upload them separately."
            )

        rows, invalid = [], 0
        seen = set()
        total = doc.page_count
        for page_idx in range(total):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(PDF_DPI_SCALE, PDF_DPI_SCALE), alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            rgb = np.asarray(image)
            bgr = rgb[:, :, ::-1].copy()
            try:
                results = detector.predict(bgr, imgsz=960)
                detections, _unmatched = detector.to_fen(results)
            except Exception:
                app.logger.exception("2d-chess-ocr inference failed on PDF page %s", page_idx + 1)
                invalid += 1
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id].update(progress=int((page_idx + 1) * 100 / total), current_page=page_idx + 1)
                continue

            for board_idx, det in enumerate(detections):
                placement = str(getattr(det, "fen", "") or "").strip()
                if not placement or len(placement.split("/")) != 8:
                    invalid += 1
                    continue
                turn, turn_conf = detector_turn(det)
                # The detector supplies piece placement and inferred turn, not full game history.
                fen = f"{placement} {turn} - - 0 1"
                try:
                    board = chess.Board(fen)
                    if not board.is_valid() or board.is_game_over():
                        invalid += 1
                        continue
                except ValueError:
                    invalid += 1
                    continue
                key = (board.board_fen(), board.turn)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "id": f"p{page_idx + 1}-b{board_idx + 1}-{uuid.uuid4().hex[:8]}",
                    "fen": board.fen(),
                    "fen_piece_placement": placement,
                    "turn": turn,
                    "turn_conf": round(turn_conf, 3),
                    "orientation": str(getattr(det, "orientation", "unknown") or "unknown"),
                    "orientation_conf": round(float(getattr(det, "orientation_conf", 0.0) or 0.0), 3),
                    "board_conf": round(float(getattr(det, "board_conf", 0.0) or 0.0), 3),
                    "page": page_idx + 1,
                    "board_index": board_idx + 1,
                })
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id].update(progress=int((page_idx + 1) * 100 / total), current_page=page_idx + 1)
        doc.close()
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(
                    status="done", progress=100, positions=rows,
                    rejected=invalid,
                    message=(f"Extracted {len(rows)} unique valid positions from {total} PDF pages. "
                             f"{invalid} detections were skipped because they were invalid or failed inference."),
                )
    except Exception as exc:
        app.logger.exception("PDF processing failed")
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(status="error", progress=100, error=str(exc))


def score_cp(info, color):
    score = info["score"].pov(color).score(mate_score=100000)
    return -100000 if score is None else score


def current_game():
    data = session.get("game")
    if not data:
        return None
    try:
        board = chess.Board(data["fen"])
        return {**data, "board": board}
    except Exception:
        session.pop("game", None)
        return None


def save_game(state):
    packed = {k: v for k, v in state.items() if k != "board"}
    packed["fen"] = state["board"].fen()
    session["game"] = packed
    session.modified = True


def game_response(state, ok=True):
    b = state["board"]
    return {
        "ok": ok,
        "fen": b.fen(),
        "orientation": state.get("player_color", "white"),
        "player_color": state.get("player_color", "white").title(),
        "page": state.get("page", ""),
        "board_index": state.get("board_index", ""),
        "plies": state.get("plies", 0),
        "target_plies": TARGET_PLIES,
        "finished": bool(state.get("finished", False)),
        "completed": bool(state.get("completed", False)),
        "message": state.get("message", ""),
        "line": state.get("line", []),
        "stats": {
            "completed": int(bool(state.get("completed", False))),
            "attempts": int(state.get("attempts", 0)),
            "mistakes": int(state.get("mistakes", 0)),
        },
    }


@app.get("/health")
def health():
    return jsonify(ok=True, service="chess-trainer", database=False)


@app.get("/")
def home():
    return render_template_string(HTML)


@app.post("/api/upload-pdf")
def upload_pdf():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify(ok=False, message="Choose a PDF file first."), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify(ok=False, message="Please upload a PDF file."), 400
    payload = uploaded.read()
    if not payload:
        return jsonify(ok=False, message="The uploaded file is empty."), 400
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "processing", "progress": 0, "current_page": 0,
            "message": "Starting the 2d-chess-ocr model…",
        }
    EXECUTOR.submit(process_pdf, payload, job_id)
    return jsonify(ok=True, job_id=job_id)


@app.get("/api/job/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify(ok=False, message="Job not found. Upload the PDF again."), 404
        return jsonify(ok=True, **job)


@app.post("/api/new")
def new_puzzle():
    data = request.get_json(silent=True) or {}
    fen = str(data.get("fen", ""))
    mode = str(data.get("mode", "Diagram"))
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify(ok=False, message="Invalid FEN for this position."), 400
    if not board.is_valid() or board.is_game_over():
        return jsonify(ok=False, message="This position is not a valid playable chess position."), 400
    wanted = {"White": chess.WHITE, "Black": chess.BLACK}.get(mode)
    if wanted is not None and board.turn != wanted:
        board.turn = wanted
        if not board.is_valid() or board.is_game_over():
            return jsonify(ok=False, message="This position cannot legally be used with that side to move."), 400
    elif mode == "Random":
        alternate = board.copy(stack=False)
        alternate.turn = not alternate.turn
        choices = [board] + ([alternate] if alternate.is_valid() and not alternate.is_game_over() else [])
        board = random.choice(choices)
    player_color = "white" if board.turn == chess.WHITE else "black"
    state = {
        "fen": board.fen(), "player_color": player_color,
        "start_fen": board.fen(), "page": str(data.get("page", "")),
        "board_index": str(data.get("board_index", "")), "plies": 0,
        "line": [], "attempts": 0, "mistakes": 0,
        "finished": False, "completed": False,
        "message": f"You play {player_color.title()}. Find a strong move. Stockfish will reply automatically.",
    }
    state["board"] = board
    save_game(state)
    return jsonify(game_response(state))


@app.get("/api/resume")
def resume_game():
    state = current_game()
    if state is None:
        return jsonify(ok=False, message="Choose a puzzle and press Start puzzle."), 404
    return jsonify(game_response(state))


@app.post("/api/switch-side")
def switch_side():
    state = current_game()
    if state is None:
        return jsonify(ok=False, message="Start a puzzle first."), 400
    if state["finished"]:
        return jsonify(ok=False, message="Start another puzzle before switching sides."), 400
    board = state["board"].copy(stack=False)
    board.turn = not board.turn
    if not board.is_valid():
        return jsonify(ok=False, message="Changing side to move makes this position invalid."), 400
    state.update({
        "board": board, "fen": board.fen(),
        "player_color": "white" if board.turn == chess.WHITE else "black",
        "start_fen": board.fen(), "plies": 0, "line": [], "attempts": 0,
        "mistakes": 0, "finished": False, "completed": False,
        "message": "Side changed. This side-to-move variation has not been checked against a book solution.",
    })
    save_game(state)
    return jsonify(game_response(state))


@app.post("/api/move")
def submit_move():
    state = current_game()
    if state is None:
        return jsonify(ok=False, message="Start a puzzle first."), 400
    board = state["board"]
    if state["finished"]:
        return jsonify(ok=False, message="This puzzle is finished. Start the next one.", fen=board.fen()), 400
    player_color = chess.WHITE if state["player_color"] == "white" else chess.BLACK
    if board.turn != player_color:
        return jsonify(ok=False, message="Wait for the engine reply.", fen=board.fen()), 400
    data = request.get_json(silent=True) or {}
    try:
        move = chess.Move.from_uci(str(data.get("uci", "")))
    except ValueError:
        return jsonify(ok=False, message="Invalid move.", fen=board.fen()), 400
    if move not in board.legal_moves:
        return jsonify(ok=False, message="Illegal move.", fen=board.fen()), 400
    try:
        with ENGINE_LOCK:
            engine = get_engine()
            legal_count = board.legal_moves.count()
            infos = engine.analyse(board, chess.engine.Limit(time=ENGINE_TIME), multipv=max(1, min(MULTIPV, legal_count)))
            if isinstance(infos, dict):
                infos = [infos]
            scored = [(score_cp(info, board.turn), info["pv"][0]) for info in infos if info.get("pv")]
            if not scored:
                raise RuntimeError("Stockfish returned no candidate moves.")
            best_score = max(score for score, _ in scored)
            selected_info = next((info for info in infos if info.get("pv") and info["pv"][0] == move), None)
            if selected_info is None:
                selected_info = engine.analyse(board, chess.engine.Limit(time=ENGINE_TIME), root_moves=[move])
                if isinstance(selected_info, list):
                    selected_info = selected_info[0]
            loss_cp = max(0, best_score - score_cp(selected_info, board.turn))
            if loss_cp > ACCEPTABLE_LOSS_CP:
                state["attempts"] += 1
                state["mistakes"] += 1
                alternatives = [board.san(m) for score, m in sorted(scored, key=lambda x: x[0], reverse=True)
                                if score >= best_score - ACCEPTABLE_LOSS_CP]
                state["message"] = "Try again. Strong alternatives: " + ", ".join(alternatives[:6]) + "."
                save_game(state)
                return jsonify(game_response(state, ok=False))
            san = board.san(move)
            board.push(move)
            state["plies"] += 1
            state["attempts"] += 1
            state["line"].append({"by": "You", "san": san})
            if board.is_game_over():
                state["finished"] = True
                state["completed"] = True
                state["message"] = "The game has ended. Puzzle complete."
            elif state["plies"] >= TARGET_PLIES:
                state["finished"] = True
                state["completed"] = True
                state["message"] = "Training line complete."
            else:
                reply = engine.play(board, chess.engine.Limit(time=ENGINE_TIME)).move
                if reply is None:
                    state["finished"] = True
                    state["completed"] = True
                    state["message"] = "No engine reply available."
                else:
                    reply_san = board.san(reply)
                    board.push(reply)
                    state["plies"] += 1
                    state["line"].append({"by": "Stockfish", "san": reply_san})
                    if board.is_game_over():
                        state["finished"] = True
                        state["completed"] = True
                        state["message"] = "The game has ended. Puzzle complete."
                    elif state["plies"] >= TARGET_PLIES:
                        state["finished"] = True
                        state["completed"] = True
                        state["message"] = "Training line complete."
                    else:
                        state["message"] = f"Accepted {san}. Stockfish replied {reply_san}. Your move."
            state["fen"] = board.fen()
            save_game(state)
        return jsonify(game_response(state))
    except Exception as exc:
        app.logger.exception("Move processing failed")
        return jsonify(ok=False, message=f"Engine error: {exc}", fen=board.fen()), 500


@app.post("/api/hint")
def hint():
    state = current_game()
    if state is None or state["finished"]:
        return jsonify(ok=False, message="Start an active puzzle first."), 400
    board = state["board"]
    try:
        with ENGINE_LOCK:
            engine = get_engine()
            infos = engine.analyse(board, chess.engine.Limit(time=ENGINE_TIME), multipv=max(1, min(MULTIPV, board.legal_moves.count())))
        if isinstance(infos, dict):
            infos = [infos]
        scored = [(score_cp(i, board.turn), board.san(i["pv"][0])) for i in infos if i.get("pv")]
        if not scored:
            raise RuntimeError("No hint is available.")
        best_score = max(score for score, _ in scored)
        alternatives = [san for score, san in sorted(scored, key=lambda x: x[0], reverse=True)
                        if score >= best_score - ACCEPTABLE_LOSS_CP]
        state["message"] = "Hint: " + alternatives[0] + ". Other strong options: " + ", ".join(alternatives[1:6]) + "."
        save_game(state)
        response = game_response(state)
        response["hint"] = state["message"]
        return jsonify(response)
    except Exception as exc:
        app.logger.exception("Hint failed")
        return jsonify(ok=False, message=f"Hint error: {exc}"), 500


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess Tactics Trainer</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f4f0;color:#202720;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1100px;margin:auto;padding:20px 13px}h1{font-size:clamp(27px,5vw,40px);margin:0 0 6px}h2{font-size:17px;margin:0 0 9px}p{margin:5px 0}.muted{color:#70786f;font-size:13px;line-height:1.5}.panel{background:white;border:1px solid #e0e3dd;border-radius:15px;padding:16px;margin-bottom:15px;box-shadow:0 3px 15px #17251b08}.layout{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,.95fr);gap:15px;align-items:start}button,input,select{font:inherit;border-radius:9px;padding:10px 12px}input,select{border:1px solid #d9dfd7;background:white;max-width:100%}button{border:0;font-weight:700;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}.primary{background:#285440;color:white}.secondary{background:#eaf0e9;color:#244532}.wide{width:100%}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.row input{flex:1;min-width:160px}.status{padding:12px;border-radius:9px;background:#f3f5f1;font-size:13px;line-height:1.5;overflow-wrap:anywhere}.error{background:#fff0ed;color:#8d3128}#board{width:min(100%,520px);margin:15px auto}.boardhead{display:flex;align-items:center;justify-content:space-between;gap:8px}.tag{padding:6px 9px;background:#edf2eb;border-radius:8px;font-size:12px}.actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}.actions button{flex:1;min-width:95px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:13px 0}.stat{border:1px solid #e1e4df;border-radius:10px;padding:10px}.stat strong{font-size:21px;display:block}.stat span,.small{color:#727970;font-size:11px;line-height:1.5}.small{margin-top:12px}.hidden{display:none!important}label{display:block;font-size:12px;font-weight:700;margin:9px 0 5px}#moves{padding-left:20px;font-size:12px;line-height:1.65}.positions{display:grid;gap:8px;max-height:290px;overflow:auto}.position{display:flex;justify-content:space-between;gap:8px;border:1px solid #e2e6df;border-radius:9px;padding:9px;font-size:12px;align-items:center}.position button{padding:6px 8px}.position.current{border-color:#285440;background:#f0f7f1}.formhint{font-size:12px;color:#70786f}@media(max-width:720px){main{padding:12px 8px}.layout{grid-template-columns:1fr}.panel{padding:13px}.stats{grid-template-columns:repeat(3,1fr)}}
</style></head><body><main>
<header><p class="muted">2D-CHESS-OCR · STOCKFISH · NO DATABASE</p><h1>Chess Tactics Trainer</h1><p class="muted">Upload a tactics book PDF. Recognized positions feed directly into the interactive trainer.</p></header>
<section class="panel"><h2>1. Upload a PDF book</h2><div class="row" style="margin-top:10px"><input id="pdf" type="file" accept="application/pdf,.pdf"><button id="upload" class="primary">Extract positions</button></div><p class="formhint">Uses the original AndrewSpano/2d-chess-ocr trained model. Scanned books are supported as page images, but recognition may miss or misread diagrams.</p><div id="uploadStatus" class="status" style="margin-top:10px">Choose a PDF to begin.</div><div id="positionCount" class="muted" style="margin-top:8px"></div><div id="positions" class="positions" style="margin-top:9px"></div></section>
<div class="layout"><section class="panel"><div class="boardhead"><div><h2>2. Solve a position</h2><p id="page" class="muted">Upload a PDF first.</p></div><span id="turn" class="tag">Waiting</span></div><div id="board"></div><div class="actions"><button id="hint" class="secondary" disabled>Hint</button><button id="switchSide" class="secondary" disabled>Switch colour</button><button id="next" class="primary" disabled>Next puzzle</button></div><label for="mode">Side to play</label><select id="mode" class="wide"><option value="Diagram">Use diagram's inferred turn</option><option value="White">Play White</option><option value="Black">Play Black</option><option value="Random" selected>Random colour</option></select><button id="start" class="primary wide" style="margin-top:8px" disabled>Start selected/random puzzle</button><p class="small">Moves are assessed with Stockfish. Other strong moves may be accepted; book answer keys are not used.</p></section>
<aside class="panel"><h2>Current puzzle</h2><div id="status" class="status">Upload a PDF and select a position.</div><div class="stats"><div class="stat"><strong id="plies">0 / 10</strong><span>Half-moves</span></div><div class="stat"><strong id="attempts">0</strong><span>Attempts</span></div><div class="stat"><strong id="mistakes">0</strong><span>Rejected moves</span></div></div><h2>Move history</h2><ol id="moves"><li>Moves will appear here.</li></ol><p class="small">No account or database is used. Your extracted position list is kept in this browser for convenience; a Render restart clears the current game session.</p></aside></div>
<footer class="muted">OCR model: <a href="https://github.com/AndrewSpano/2d-chess-ocr" target="_blank" rel="noopener">AndrewSpano/2d-chess-ocr</a>. This application uses CPU inference by default; large books may take time.</footer>
</main><script>
(()=>{const $=id=>document.getElementById(id);let board=null,positions=[],selected=0,loaded=false,busy=false,playing=false,playerColor='White',jobTimer=null;
function status(m,error=false){$('status').textContent=m||'';$('status').className=error?'status error':'status'}
function uploadStatus(m,error=false){$('uploadStatus').textContent=m||'';$('uploadStatus').className=error?'status error':'status'}
function post(url,body){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}).then(async r=>({response:r,data:await r.json()}))}
function storePositions(){try{localStorage.setItem('chessTrainerPositions',JSON.stringify(positions))}catch(e){uploadStatus('Extracted positions are too large for browser storage. Keep this page open and use the list below.',true)}}
function showPositions(){const list=$('positions');list.replaceChildren();$('positionCount').textContent=positions.length?`${positions.length} valid candidate positions · click any row to select it`:'No positions loaded.';positions.forEach((p,i)=>{const row=document.createElement('div');row.className='position'+(i===selected?' current':'');const info=document.createElement('span');info.textContent=`#${i+1} · page ${p.page} · board ${p.board_index} · ${p.turn==='w'?'White':'Black'} to move · turn conf ${Math.round((p.turn_conf||0)*100)}%`;const b=document.createElement('button');b.className='secondary';b.textContent='Select';b.onclick=()=>{selected=i;showPositions();status(`Selected page ${p.page}, board ${p.board_index}. Press Start puzzle.`);$('start').disabled=false};row.append(info,b);list.appendChild(row)});loaded=positions.length>0;$('start').disabled=!loaded;$('next').disabled=!loaded}
function render(d){if(d.fen&&board){board.orientation(d.orientation||'white');board.position(d.fen,false)}playerColor=d.player_color||playerColor;playing=!d.finished;busy=false;$('page').textContent=d.page?`PDF page ${d.page} · board ${d.board_index}`:'Source page not recorded';$('turn').textContent=d.finished?'Finished':`You play ${playerColor}`;status(d.message||'');$('plies').textContent=`${d.plies} / ${d.target_plies}`;$('attempts').textContent=d.stats?.attempts??0;$('mistakes').textContent=d.stats?.mistakes??0;const ol=$('moves');ol.replaceChildren();if(!d.line||!d.line.length){const li=document.createElement('li');li.textContent='Make your first move.';ol.appendChild(li)}else d.line.forEach(m=>{const li=document.createElement('li');li.textContent=`${m.by}: ${m.san}`;ol.appendChild(li)});$('hint').disabled=!playing||busy;$('switchSide').disabled=!playing||busy;$('next').disabled=!loaded||busy;$('start').disabled=!loaded||busy}
async function newPuzzle(randomChoice=false){if(!loaded||busy)return;busy=true;$('start').disabled=$('next').disabled=$('hint').disabled=$('switchSide').disabled=true;status('Starting puzzle…');let item=positions[selected]||positions[0];if(randomChoice&&positions.length>1){const choices=positions.map((p,i)=>i).filter(i=>i!==selected);selected=choices[Math.floor(Math.random()*choices.length)];showPositions();item=positions[selected]}const mode=$('mode').value;try{const {response,data}=await post('/api/new',{fen:item.fen,page:item.page,board_index:item.board_index,mode});if(!response.ok||!data.ok)throw Error(data.message||'Could not start puzzle.');render(data)}catch(e){busy=false;status(e.message,true);$('start').disabled=$('next').disabled=!loaded}}
async function pollJob(id){try{const r=await fetch('/api/job/'+id);const d=await r.json();if(!r.ok||!d.ok)throw Error(d.message||'Processing job failed');if(d.status==='processing'){uploadStatus(`Processing page ${d.current_page||0} · ${d.progress||0}% — loading and running 2d-chess-ocr`);jobTimer=setTimeout(()=>pollJob(id),1000);return}if(d.status==='error')throw Error(d.error||'PDF processing failed');positions=d.positions||[];storePositions();selected=0;showPositions();uploadStatus(d.message||`Extracted ${positions.length} positions.`);if(positions.length){await newPuzzle()}else uploadStatus((d.message||'No valid positions found.')+' Try a sharper or smaller PDF.',true)}catch(e){uploadStatus('Extraction error: '+e.message,true)}finally{$('upload').disabled=false}}
async function uploadPDF(){const file=$('pdf').files[0];if(!file){uploadStatus('Choose a PDF first.',true);return}if(!file.name.toLowerCase().endsWith('.pdf')){uploadStatus('Please choose a PDF file.',true);return}clearTimeout(jobTimer);$('upload').disabled=true;uploadStatus('Uploading PDF…');try{const form=new FormData();form.append('file',file);const r=await fetch('/api/upload-pdf',{method:'POST',body:form});const d=await r.json();if(!r.ok||!d.ok)throw Error(d.message||'Upload failed');uploadStatus('PDF accepted. Preparing the OCR model…');pollJob(d.job_id)}catch(e){$('upload').disabled=false;uploadStatus('Upload error: '+e.message,true)}}
async function makeMove(uci){if(!playing||busy)return;busy=true;$('hint').disabled=$('next').disabled=$('switchSide').disabled=true;try{const {response,data}=await post('/api/move',{uci});if(data.fen&&board)board.position(data.fen,false);if(!response.ok||!data.ok){busy=false;if(data.fen)render(data);status(data.message||'Move rejected. Try again.',true);return}render(data)}catch(e){busy=false;status('Move request failed: '+e.message,true)}}
async function askHint(){if(busy||!playing)return;busy=true;$('hint').disabled=true;try{const {response,data}=await post('/api/hint',{});if(!response.ok||!data.ok)throw Error(data.message||'No hint available');render(data);status(data.hint||data.message)}catch(e){busy=false;status(e.message,true)}}
async function switchSide(){if(busy||!playing)return;busy=true;try{const {response,data}=await post('/api/switch-side',{});if(!response.ok||!data.ok)throw Error(data.message||'Could not switch side');render(data)}catch(e){busy=false;status(e.message,true)}}
function onDragStart(source,piece){if(!playing||busy)return false;return piece[0]===(playerColor==='White'?'w':'b')}
function onDrop(source,target,piece){if(!playing||busy)return 'snapback';if(!target||target==='offboard'||source===target)return 'snapback';const prom=piece[1]==='P'&&((piece[0]==='w'&&target[1]==='8')||(piece[0]==='b'&&target[1]==='1'))?'q':'';makeMove(source+target+prom)}
$('upload').addEventListener('click',uploadPDF);$('start').addEventListener('click',newPuzzle);$('next').addEventListener('click',()=>newPuzzle(true));$('hint').addEventListener('click',askHint);$('switchSide').addEventListener('click',switchSide);
if(typeof Chessboard!=='function'){uploadStatus('Could not load the chessboard library. Check your internet connection.',true);return}board=Chessboard('board',{position:'start',draggable:true,pieceTheme:'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',onDragStart,onDrop});window.addEventListener('resize',()=>board.resize());try{const saved=JSON.parse(localStorage.getItem('chessTrainerPositions')||'[]');if(Array.isArray(saved)&&saved.length){positions=saved;showPositions();uploadStatus(`Loaded ${positions.length} positions from this browser.`)}}catch(e){}fetch('/api/resume').then(async r=>{if(r.ok)render(await r.json())}).catch(()=>{});})();
</script></body></html>'''


@app.errorhandler(413)
def too_large(_exc):
    return jsonify(ok=False, message=f"PDF is too large. Current upload limit: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."), 413


if __name__ == "__main__":
    print("Chess Tactics Trainer — no database")
    print("Stockfish:", find_stockfish() or "not found yet")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
