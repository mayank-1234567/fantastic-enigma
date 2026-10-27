import csv
import io
import json
import os
import random
import shutil
import threading
from pathlib import Path
from datetime import datetime, timezone

import chess
import chess.engine
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# ---------------- CONFIGURATION ----------------

TARGET_PLIES = 10
ENGINE_TIME = 0.35
ACCEPTABLE_LOSS_CP = 125
MULTIPV = 8

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = DATA_DIR / "progress.json"

POSITIONS = []
SESSIONS = {}

ENGINE = None
ENGINE_LOCK = threading.Lock()


# ---------------- STOCKFISH ----------------

def find_stockfish():
    candidates = [
        os.environ.get("STOCKFISH_PATH"),
        shutil.which("stockfish"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\Stockfish\stockfish.exe",
    ]

    for path in candidates:
        if not path:
            continue

        p = Path(path)
        if p.is_file() and (
            os.access(str(p), os.X_OK)
            or str(p).lower().endswith(".exe")
        ):
            return str(p)

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
        raise RuntimeError(
            "Stockfish was not found on this computer. "
            "Install Stockfish and set STOCKFISH_PATH "
            "to its executable path."
        )

    ENGINE = chess.engine.SimpleEngine.popen_uci(path, timeout=20)

    options = {}

    if "Threads" in ENGINE.options:
        options["Threads"] = max(1, min(4, os.cpu_count() or 2))

    if "Hash" in ENGINE.options:
        options["Hash"] = 128

    if options:
        ENGINE.configure(options)

    return ENGINE


def score_cp(info, color):
    score = info["score"].pov(color).score(mate_score=100000)
    return -100000 if score is None else score


# ---------------- IMPORT CSV ----------------

def parse_turn(value):
    value = str(value or "").strip().lower()

    if value in ("w", "white", "true", "1"):
        return chess.WHITE

    if value in ("b", "black", "false", "0"):
        return chess.BLACK

    return None


def make_boards(value, turn_hint=None):
    """Accept full FEN or piece placement from OCR."""

    value = str(value or "").strip()

    if not value:
        return []

    parts = value.split()

    try:
        # A FEN with an explicit side to move.
        if len(parts) >= 2 and parts[1] in ("w", "b"):
            if len(parts) >= 6:
                fen = " ".join(parts[:6])
            else:
                fen = f"{parts[0]} {parts[1]} - - 0 1"

            board = chess.Board(fen)
            return [board] if board.is_valid() else []

        # Piece placement only: use the OCR turn when available.
        placement = parts[0]

        colors = (
            [turn_hint]
            if turn_hint is not None
            else [chess.WHITE, chess.BLACK]
        )

        boards = []

        for color in colors:
            turn = "w" if color == chess.WHITE else "b"
            board = chess.Board(
                f"{placement} {turn} - - 0 1"
            )

            if board.is_valid():
                boards.append(board)

        return boards

    except (ValueError, IndexError):
        return []


def parse_csv(content):
    reader = csv.DictReader(io.StringIO(content))

    if not reader.fieldnames:
        raise ValueError("The CSV has no header row.")

    positions = []
    seen = set()
    invalid = 0

    for raw_row in reader:
        row = {
            str(key).strip().lower(): value
            for key, value in raw_row.items()
            if key is not None
        }

        value = (
            row.get("fen")
            or row.get("provisional_fen")
            or row.get("fen_piece_placement")
            or row.get("piece_placement")
            or row.get("placement")
        )

        turn_hint = parse_turn(
            row.get("turn") or row.get("side_to_move")
        )

        boards = make_boards(value, turn_hint)

        if not boards:
            invalid += 1
            continue

        for board in boards:
            fen = board.fen()

            if fen in seen or board.is_game_over():
                continue

            seen.add(fen)

            positions.append({
                "fen": fen,
                "page": row.get("page", "") or "",
                "board_index": row.get("board_index", "") or "",
            })

    if not positions:
        raise ValueError(
            "No valid chess positions were found. Upload "
            "puzzle_candidates.csv or stockfish_analysis.csv "
            "containing a FEN column."
        )

    return positions, invalid


# ---------------- PROGRESS SAVING ----------------

def load_progress():
    try:
        result = json.loads(
            PROGRESS_FILE.read_text(encoding="utf-8")
        )
        return result if isinstance(result, list) else []

    except (OSError, json.JSONDecodeError):
        return []


def get_stats(state=None):
    records = load_progress()

    stats = {
        "completed": sum(
            bool(item.get("completed")) for item in records
        ),
        "attempts": sum(
            int(item.get("attempts", 0)) for item in records
        ),
        "mistakes": sum(
            int(item.get("mistakes", 0)) for item in records
        ),
    }

    # Don't count a finished puzzle twice.
    if state is not None and not state["saved"]:
        stats["attempts"] += state["attempts"]
        stats["mistakes"] += state["mistakes"]

    return stats


def finish_puzzle(state, message):
    state["finished"] = True
    state["message"] = message

    if state["saved"]:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "page": state["page"],
        "side": (
            "White"
            if state["player_color"] == chess.WHITE
            else "Black"
        ),
        "start_fen": state["start_fen"],
        "final_fen": state["board"].fen(),
        "plies": state["plies"],
        "moves": state["line"],
        "attempts": state["attempts"],
        "mistakes": state["mistakes"],
        "completed": state["completed"],
    }

    records = load_progress()
    records.append(record)

    PROGRESS_FILE.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    state["saved"] = True


def state_response(state):
    board = state["board"]

    return {
        "ok": True,
        "fen": board.fen(),
        "orientation": (
            "white"
            if state["player_color"] == chess.WHITE
            else "black"
        ),
        "player_color": (
            "White"
            if state["player_color"] == chess.WHITE
            else "Black"
        ),
        "page": state["page"],
        "plies": state["plies"],
        "target_plies": TARGET_PLIES,
        "finished": state["finished"],
        "completed": state["completed"],
        "message": state["message"],
        "line": state["line"],
        "stats": get_stats(state),
    }


# ---------------- WEB ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    global POSITIONS

    uploaded = request.files.get("file")

    if not uploaded or not uploaded.filename:
        return jsonify(
            ok=False,
            message="Choose a CSV file first.",
        ), 400

    try:
        content = uploaded.read().decode(
            "utf-8-sig", errors="replace"
        )

        positions, invalid = parse_csv(content)

    except Exception as exc:
        return jsonify(ok=False, message=str(exc)), 400

    POSITIONS = positions

    whites = sum(
        chess.Board(item["fen"]).turn == chess.WHITE
        for item in POSITIONS
    )

    blacks = len(POSITIONS) - whites

    return jsonify(
        ok=True,
        count=len(POSITIONS),
        white=whites,
        black=blacks,
        invalid=invalid,
        message=f"Loaded {len(POSITIONS)} legal positions.",
    )


@app.route("/api/new", methods=["POST"])
def new_puzzle():
    if not POSITIONS:
        return jsonify(
            ok=False,
            message="Upload your candidate CSV first.",
        ), 400

    payload = request.get_json(silent=True) or {}

    session_id = str(payload.get("session_id", ""))
    side = str(payload.get("side", "Random")).title()

    if side not in ("White", "Black", "Random"):
        side = "Random"

    pool = POSITIONS

    if side == "White":
        pool = [
            item for item in POSITIONS
            if chess.Board(item["fen"]).turn == chess.WHITE
        ]

    elif side == "Black":
        pool = [
            item for item in POSITIONS
            if chess.Board(item["fen"]).turn == chess.BLACK
        ]

    prefix = ""

    if not pool:
        pool = POSITIONS
        prefix = (
            f"No {side}-to-move positions were available. "
        )

    previous = SESSIONS.get(session_id)
    previous_fen = (
        previous["start_fen"] if previous else None
    )

    fresh = [
        item for item in pool
        if item["fen"] != previous_fen
    ]

    if fresh:
        pool = fresh

    item = random.choice(pool)
    board = chess.Board(item["fen"])

    color_name = "White" if board.turn else "Black"

    state = {
        "board": board,
        "player_color": board.turn,
        "start_fen": board.fen(),
        "page": item["page"],
        "board_index": item["board_index"],
        "plies": 0,
        "line": [],
        "attempts": 0,
        "mistakes": 0,
        "finished": False,
        "completed": False,
        "saved": False,
        "message": (
            f"{prefix}You play {color_name}. Find a strong move."
        ),
    }

    SESSIONS[session_id] = state

    return jsonify(state_response(state))


@app.route("/api/state")
def current_state():
    session_id = request.args.get("session_id", "")
    state = SESSIONS.get(session_id)

    if state is None:
        return jsonify(
            ok=False,
            message="No active puzzle.",
        ), 404

    return jsonify(state_response(state))


@app.route("/api/move", methods=["POST"])
def submit_move():
    payload = request.get_json(silent=True) or {}

    session_id = str(payload.get("session_id", ""))
    uci = str(payload.get("uci", ""))

    state = SESSIONS.get(session_id)

    if state is None:
        return jsonify(
            ok=False,
            message="Start a puzzle first.",
        ), 400

    board = state["board"]

    if state["finished"]:
        return jsonify(
            ok=False,
            message="Puzzle finished. Start the next puzzle.",
            fen=board.fen(),
        ), 400

    if board.turn != state["player_color"]:
        return jsonify(
            ok=False,
            message="Wait for Stockfish's reply.",
            fen=board.fen(),
        ), 400

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return jsonify(
            ok=False,
            message="Invalid move.",
            fen=board.fen(),
        ), 400

    if move not in board.legal_moves:
        return jsonify(
            ok=False,
            message="That move is illegal.",
            fen=board.fen(),
        ), 400

    try:
        with ENGINE_LOCK:
            engine = get_engine()

            infos = engine.analyse(
                board,
                chess.engine.Limit(time=ENGINE_TIME),
                multipv=min(
                    MULTIPV,
                    board.legal_moves.count(),
                ),
            )

            if isinstance(infos, dict):
                infos = [infos]

            scored = [
                (score_cp(info, board.turn), info["pv"][0])
                for info in infos
                if info.get("pv")
            ]

            if not scored:
                raise RuntimeError(
                    "Stockfish returned no candidate moves."
                )

            best_score = max(score for score, _ in scored)

            selected_info = next(
                (
                    info for info in infos
                    if info.get("pv")
                    and info["pv"][0] == move
                ),
                None,
            )

            # Analyze a move directly if it is outside MultiPV.
            if selected_info is None:
                selected_info = engine.analyse(
                    board,
                    chess.engine.Limit(time=ENGINE_TIME),
                    root_moves=[move],
                )

                if isinstance(selected_info, list):
                    selected_info = selected_info[0]

            selected_score = score_cp(
                selected_info, board.turn
            )

            loss_cp = max(0, best_score - selected_score)

            if loss_cp > ACCEPTABLE_LOSS_CP:
                state["attempts"] += 1
                state["mistakes"] += 1

                alternatives = [
                    board.san(candidate)
                    for score, candidate in sorted(
                        scored,
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    if score >= best_score - ACCEPTABLE_LOSS_CP
                ]

                state["message"] = (
                    "Try again. Strong alternatives: "
                    + ", ".join(alternatives[:6])
                    + "."
                )

                response = state_response(state)
                response["ok"] = False
                response["fen"] = board.fen()

                return jsonify(response)

            # Accept the user's move.
            state["attempts"] += 1

            user_san = board.san(move)
            board.push(move)

            state["plies"] += 1
            state["line"].append({
                "by": "You",
                "san": user_san,
            })

            if board.is_game_over():
                state["completed"] = True

                finish_puzzle(
                    state,
                    "Game ended. Progress saved.",
                )

                return jsonify(state_response(state))

            if state["plies"] >= TARGET_PLIES:
                state["completed"] = True

                finish_puzzle(
                    state,
                    "Puzzle line complete. Progress saved.",
                )

                return jsonify(state_response(state))

            # Stockfish's automatic reply.
            reply = engine.play(
                board,
                chess.engine.Limit(time=ENGINE_TIME),
            ).move

            if reply is None:
                state["completed"] = True

                finish_puzzle(
                    state,
                    "No engine reply available. Progress saved.",
                )

                return jsonify(state_response(state))

            reply_san = board.san(reply)
            board.push(reply)

            state["plies"] += 1
            state["line"].append({
                "by": "Stockfish",
                "san": reply_san,
            })

        if board.is_game_over():
            state["completed"] = True

            finish_puzzle(
                state,
                "Game ended. Progress saved.",
            )

        elif state["plies"] >= TARGET_PLIES:
            state["completed"] = True

            finish_puzzle(
                state,
                "Puzzle line complete. Progress saved.",
            )

        else:
            state["message"] = (
                f"Accepted {user_san}. "
                f"Stockfish replied {reply_san}. Your move."
            )

        return jsonify(state_response(state))

    except Exception as exc:
        app.logger.exception("Move processing failed")

        return jsonify(
            ok=False,
            message=f"Stockfish error: {exc}",
            fen=board.fen(),
        ), 500


@app.route("/api/hint", methods=["POST"])
def hint():
    payload = request.get_json(silent=True) or {}

    state = SESSIONS.get(
        str(payload.get("session_id", ""))
    )

    if state is None:
        return jsonify(
            ok=False,
            message="Start a puzzle first.",
        ), 400

    if state["finished"]:
        return jsonify(
            ok=False,
            message="This puzzle is finished.",
        ), 400

    board = state["board"]

    try:
        with ENGINE_LOCK:
            engine = get_engine()

            infos = engine.analyse(
                board,
                chess.engine.Limit(time=ENGINE_TIME),
                multipv=min(
                    MULTIPV,
                    board.legal_moves.count(),
                ),
            )

        if isinstance(infos, dict):
            infos = [infos]

        scored = [
            (score_cp(info, board.turn), board.san(info["pv"][0]))
            for info in infos
            if info.get("pv")
        ]

        if not scored:
            raise RuntimeError("No hint available.")

        best_score = max(score for score, _ in scored)

        alternatives = [
            san
            for score, san in sorted(
                scored,
                key=lambda item: item[0],
                reverse=True,
            )
            if score >= best_score - ACCEPTABLE_LOSS_CP
        ]

        best_move = max(scored, key=lambda item: item[0])[1]

        state["message"] = (
            f"Hint: {best_move}. Other strong options: "
            + ", ".join(alternatives[:6])
            + "."
        )

        response = state_response(state)
        response["hint"] = state["message"]

        return jsonify(response)

    except Exception as exc:
        app.logger.exception("Hint failed")

        return jsonify(
            ok=False,
            message=f"Hint error: {exc}",
        ), 500


@app.route("/api/stats")
def stats():
    return jsonify(ok=True, stats=get_stats())


# ============================================================
# COMPLETE BROWSER INTERFACE
# ============================================================

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Chess Puzzle Trainer</title>

<link rel="stylesheet"
 href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #f3f4f0;
    color: #202720;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}

main {
    max-width: 1120px;
    margin: auto;
    padding: 24px 16px;
}

header { margin-bottom: 20px; }

h1 {
    margin: 0 0 6px;
    font-size: clamp(28px, 4vw, 40px);
}

h2 {
    font-size: 17px;
    margin: 0 0 8px;
}

p { margin: 5px 0; }

.muted {
    color: #727970;
    font-size: 13px;
}

.panel {
    border: 1px solid #dfe3dc;
    border-radius: 15px;
    padding: 18px;
    background: white;
    box-shadow: 0 3px 15px #17251b08;
}

.import { margin-bottom: 16px; }

.uploadline {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}

input[type=file] {
    flex: 1;
    min-width: 220px;
}

button, select { font: inherit; }

button, select {
    border-radius: 9px;
    padding: 10px 13px;
}

button {
    cursor: pointer;
    border: 0;
    font-weight: 700;
}

button:disabled {
    opacity: .45;
    cursor: not-allowed;
}

.primary {
    background: #285440;
    color: white;
}

.secondary {
    background: #eaf0e9;
    color: #244532;
}

select {
    border: 1px solid #dce0da;
    background: white;
}

.layout {
    display: grid;
    grid-template-columns: minmax(300px, 1.05fr)
                           minmax(280px, .95fr);
    gap: 16px;
}

#board {
    width: min(100%, 510px);
    margin: 12px auto;
}

.board-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
}

.tag {
    background: #eef2eb;
    border-radius: 8px;
    padding: 6px 9px;
    font-size: 12px;
    font-weight: 700;
}

.actions {
    display: flex;
    justify-content: center;
    gap: 9px;
    margin-top: 12px;
}

.status {
    padding: 13px;
    background: #f4f6f2;
    border-radius: 10px;
    line-height: 1.5;
    font-size: 14px;
    min-height: 60px;
}

.status.error {
    background: #fff0ed;
    border: 1px solid #f1d4ce;
}

.stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 13px 0 20px;
}

.stat {
    border: 1px solid #e1e4df;
    border-radius: 10px;
    padding: 11px;
}

.stat strong {
    display: block;
    font-size: 22px;
}

.stat span {
    color: #737a73;
    font-size: 11px;
}

.wide {
    width: 100%;
    margin-top: 8px;
}

label {
    display: block;
    font-size: 12px;
    font-weight: 700;
    margin: 6px 0;
}

#moves {
    padding-left: 20px;
    line-height: 1.65;
    font-size: 13px;
}

#uploadStatus {
    margin-top: 10px;
    font-size: 13px;
    color: #555e56;
}

footer {
    margin: 18px 2px 0;
    font-size: 11px;
    color: #6e766e;
    line-height: 1.6;
}

@media (max-width: 730px) {
    .layout { grid-template-columns: 1fr; }
    main { padding: 14px 10px; }
}
</style>
</head>

<body>
<main>
<header>
    <p class="muted">LOCAL MVP · STOCKFISH</p>
    <h1>Chess Puzzle Trainer</h1>
    <p class="muted">Drag pieces to play. Stockfish replies automatically.</p>
</header>

<section class="panel import">
    <h2>1. Load candidate positions</h2>
    <p class="muted">
        Upload puzzle_candidates.csv from your existing Colab work.
    </p>

    <div class="uploadline">
        <input type="file" id="csv" accept=".csv">
        <button class="primary" id="upload">Load positions</button>
    </div>

    <div id="uploadStatus">No CSV loaded yet.</div>
</section>

<div class="layout">
<section class="panel">
    <div class="board-head">
        <div>
            <h2>2. Play</h2>
            <p class="muted" id="page">Upload a CSV to begin.</p>
        </div>
        <span class="tag" id="turn">Waiting</span>
    </div>

    <div id="board"></div>

    <div class="actions">
        <button class="secondary" id="hint" disabled>Hint</button>
        <button class="primary" id="next" disabled>Next puzzle</button>
    </div>

    <p class="muted">
        Strong alternative moves may also be accepted.
    </p>
</section>

<aside class="panel">
    <h2>Session</h2>

    <div class="status" id="status">
        Upload a CSV to start.
    </div>

    <div class="stats">
        <div class="stat">
            <strong id="plies">0 / 10</strong>
            <span>Plies played</span>
        </div>
        <div class="stat">
            <strong id="completed">0</strong>
            <span>Saved completions</span>
        </div>
        <div class="stat">
            <strong id="attempts">0</strong>
            <span>Moves tried</span>
        </div>
        <div class="stat">
            <strong id="mistakes">0</strong>
            <span>Rejected moves</span>
        </div>
    </div>

    <label for="side">Side to play</label>

    <select id="side">
        <option value="Random">Random</option>
        <option value="White">White</option>
        <option value="Black">Black</option>
    </select>

    <button class="primary wide" id="start" disabled>
        Start puzzle
    </button>

    <h2 style="margin-top:22px">Move history</h2>
    <ol id="moves">
        <li>Moves will appear here.</li>
    </ol>

    <p class="muted">
        Progress is saved in data/progress.json.
    </p>
</aside>
</div>

<footer>
    The CSV is processed locally. The browser needs internet access
    to load the chessboard scripts and piece images. These OCR candidates
    have not necessarily been verified as forced tactical puzzles.
</footer>
</main>

<script>
(() => {
    const $ = id => document.getElementById(id);

    const sessionId =
        window.crypto && window.crypto.randomUUID
            ? window.crypto.randomUUID()
            : "session-" + Date.now() + "-" + Math.random();

    let board = null;
    let playing = false;
    let busy = false;
    let loaded = false;
    let playerColor = "White";

    function say(message, error = false) {
        $("status").textContent = message || "";
        $("status").className = error ? "status error" : "status";
    }

    function reportUpload(message) {
        $("uploadStatus").textContent = message;
    }

    async function postJSON(url, payload) {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload || {})
        });

        const data = await response.json();
        return { response, data };
    }

    // IMPORTANT:
    // Render the FEN's pieces into the board.
    // Do not call board.start() here; that resets the initial position.
    function showFen(fen) {
        if (!board || !fen) return;

        const position = Chessboard.fenToObj(fen);

        board.position(position, false);
    }

    function render(data) {
        if (data.fen) {
            board.orientation(data.orientation || "white");
            showFen(data.fen);
        }

        playerColor = data.player_color || playerColor;
        playing = !data.finished;
        busy = false;

        $("page").textContent = data.page
            ? "Book page " + data.page
            : "Source page not recorded";

        $("turn").textContent = data.finished
            ? "Complete"
            : "You play " + playerColor;

        say(data.message || "");

        $("plies").textContent =
            `${data.plies} / ${data.target_plies}`;

        $("completed").textContent =
            data.stats?.completed ?? 0;

        $("attempts").textContent =
            data.stats?.attempts ?? 0;

        $("mistakes").textContent =
            data.stats?.mistakes ?? 0;

        const list = $("moves");
        list.replaceChildren();

        if (!data.line || data.line.length === 0) {
            const li = document.createElement("li");
            li.textContent = "Make your first move.";
            list.appendChild(li);
        } else {
            data.line.forEach((move, index) => {
                const li = document.createElement("li");
                li.textContent =
                    `${index + 1}. ${move.by}: ${move.san}`;
                list.appendChild(li);
            });
        }

        $("hint").disabled = !playing || busy;
        $("next").disabled = !loaded || busy;
        $("start").disabled = !loaded || busy;
    }

    async function restoreBoard() {
        try {
            const response = await fetch(
                "/api/state?session_id=" + encodeURIComponent(sessionId)
            );

            if (response.ok) {
                const data = await response.json();
                render(data);
            }
        } catch (error) {
            console.error(error);
        }
    }

    async function newPuzzle() {
        if (busy || !loaded) return;

        busy = true;
        $("next").disabled = true;
        $("start").disabled = true;
        $("hint").disabled = true;

        say("Loading puzzle...");

        try {
            const { response, data } = await postJSON("/api/new", {
                session_id: sessionId,
                side: $("side").value
            });

            if (!response.ok || !data.ok) {
                throw new Error(data.message || "Could not load puzzle.");
            }

            render(data);

        } catch (error) {
            busy = false;
            say(error.message, true);

            $("next").disabled = !loaded;
            $("start").disabled = !loaded;

            console.error("New puzzle error:", error);
        }
    }

    async function submitMove(uci) {
        if (!playing || busy) return;

        busy = true;

        $("hint").disabled = true;
        $("next").disabled = true;
        $("start").disabled = true;

        try {
            const { response, data } = await postJSON("/api/move", {
                session_id: sessionId,
                uci: uci
            });

            // Both accepted and rejected moves return the server's FEN.
            // Restore the authoritative position in either case.
            if (data.fen) {
                showFen(data.fen);
            }

            if (!response.ok || data.ok === false) {
                render(data);
                say(data.message || "Move was not accepted.", true);
                return;
            }

            render(data);

        } catch (error) {
            busy = false;
            await restoreBoard();
            say("Move request failed: " + error.message, true);
            console.error("Move error:", error);

        } finally {
            $("next").disabled = !loaded;
            $("start").disabled = !loaded;
        }
    }

    async function askHint() {
        if (!playing || busy) return;

        busy = true;
        $("hint").disabled = true;

        try {
            const { response, data } = await postJSON("/api/hint", {
                session_id: sessionId
            });

            if (!response.ok || !data.ok) {
                throw new Error(data.message || "Could not get a hint.");
            }

            render(data);
            say(data.hint || data.message);

        } catch (error) {
            busy = false;
            say(error.message, true);
            console.error("Hint error:", error);

        } finally {
            $("hint").disabled = !playing;
        }
    }

    function onDragStart(source, piece) {
        if (!loaded || busy || !playing) {
            return false;
        }

        const requiredColor =
            playerColor === "White" ? "w" : "b";

        if (piece[0] !== requiredColor) {
            return false;
        }
    }

    function onDrop(source, target, piece) {
        if (!playing || busy) {
            return "snapback";
        }

        if (!target || target === "offboard" || source === target) {
            return "snapback";
        }

        // Automatically promote to a queen in this MVP.
        const isPawn = piece[1] === "P";

        const promotes =
            (piece[0] === "w" && target[1] === "8") ||
            (piece[0] === "b" && target[1] === "1");

        const promotion = isPawn && promotes ? "q" : "";

        // Server validates the move and sends the correct position back.
        submitMove(source + target + promotion);

        // Do not return "snapback" for a normal attempt:
        // submitMove will restore the correct server position.
    }

    async function uploadCsv() {
        const file = $("csv").files[0];

        if (!file) {
            reportUpload("Choose a CSV file first.");
            return;
        }

        $("upload").disabled = true;
        reportUpload("Reading CSV...");

        try {
            const form = new FormData();
            form.append("file", file);

            const response = await fetch("/api/upload", {
                method: "POST",
                body: form
            });

            const data = await response.json();

            if (!response.ok || !data.ok) {
                throw new Error(data.message || "CSV import failed.");
            }

            loaded = true;

            reportUpload(
                `Loaded ${data.count} positions. ` +
                `White to move: ${data.white}; ` +
                `Black to move: ${data.black}. ` +
                `Skipped invalid rows: ${data.invalid}.`
            );

            // Automatically start a puzzle as soon as import succeeds.
            await newPuzzle();

        } catch (error) {
            reportUpload("CSV error: " + error.message);
            say(error.message, true);
            console.error("CSV upload error:", error);

        } finally {
            $("upload").disabled = false;
        }
    }

    $("upload").addEventListener("click", uploadCsv);
    $("start").addEventListener("click", newPuzzle);
    $("next").addEventListener("click", newPuzzle);
    $("hint").addEventListener("click", askHint);

    // Check that the external board libraries loaded.
    if (typeof Chessboard !== "function") {
        reportUpload(
            "Chessboard library failed to load. Check internet access, " +
            "then refresh this page."
        );

        say("The browser could not load the chessboard library.", true);
        return;
    }

    board = Chessboard("board", {
        position: "start",
        draggable: true,
        showErrors: "console",
        pieceTheme:
            "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
        onDragStart: onDragStart,
        onDrop: onDrop
    });

    // This does NOT reset the board position.
    window.addEventListener("resize", () => board.resize());

    fetch("/api/stats")
        .then(response => response.json())
        .then(data => {
            $("completed").textContent =
                data.stats?.completed ?? 0;

            $("attempts").textContent =
                data.stats?.attempts ?? 0;

            $("mistakes").textContent =
                data.stats?.mistakes ?? 0;
        })
        .catch(error => console.error(error));
})();
</script>
</body>
</html>
"""


# ---------------- START SERVER ----------------

if __name__ == "__main__":
    print("\nCHESS PUZZLE TRAINER")
    print("--------------------")

    path = find_stockfish()

    if path:
        print("Stockfish executable:", path)
    else:
        print(
            "WARNING: Stockfish not detected. "
            "Install it and set STOCKFISH_PATH."
        )

    print("Open this address in your browser:")
    print("http://127.0.0.1:5000\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
