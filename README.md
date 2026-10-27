# Chess Tactics Trainer (Render, no database)

This combines the trainer workflow with PDF-to-candidate extraction using the original [`AndrewSpano/2d-chess-ocr`](https://github.com/AndrewSpano/2d-chess-ocr) inference API. PDF pages are rendered with PyMuPDF, then passed to `Chess2DOCRInference` from the repository's `model_inference.py`. Valid detected piece placements are sent directly into the interactive trainer; no CSV upload is required.

## Deploy to Render

1. Put the complete contents of this folder in a GitHub repository.
2. In Render, choose **New + → Blueprint** and connect that repository. Render reads `render.yaml`, which uses the Dockerfile.
3. Alternatively, choose **New + → Web Service**, choose the repository, select Docker as runtime, and deploy.
4. Wait for the Docker build. It installs Stockfish, installs Python dependencies, clones the original OCR repository, and runs its `download_models.py` to fetch the pretrained weights.
5. Open the deployed URL. Upload a PDF, wait for extraction, and start solving the positions.

## Local Docker test

```bash
docker build -t chess-tactics-trainer .
docker run --rm -p 10000:10000 -e PORT=10000 -e SECRET_KEY="replace-this" chess-tactics-trainer
```

Open `http://localhost:10000`.

## Configuration

- `MAX_PDF_MB`: upload size limit (default 25 MB)
- `MAX_PDF_PAGES`: maximum pages per upload (default 160)
- `PDF_DPI_SCALE`: PDF raster scale (default 1.65)
- `ENGINE_TIME`: Stockfish analysis time in seconds per decision (default 0.35)
- `ACCEPTABLE_LOSS_CP`: allowed centipawn loss relative to best candidate move (default 125)
- `CHESS_OCR_MODEL`: optional explicit path to the `.pt` model file
- `STOCKFISH_PATH`: optional explicit Stockfish executable path

## Important limitations

- This deployment does not use a database, login system, or server-side persistent storage.
- The extracted position list is stored in the browser's localStorage for convenience. The active game uses a signed Flask session cookie. A Render restart clears in-memory extraction jobs and active extraction tasks; it does not retain uploads.
- This first integrated version is intended for one primary user/site. The OCR job queue and model instance are in process memory and are not a multi-user shared collection service.
- Turn detection from diagrams may be uncertain. `2d-chess-ocr` infers turn from highlighted last-move tiles; some book diagrams may not include them. The UI shows turn confidence, and the side-to-play selector lets you choose White or Black.
- OCR can misread a piece or miss a diagram; this version filters invalid positions, rather than including a full manual board editor.
- The Docker image installs the original published YOLO weights during build. Verify the build log confirms `yolo26m-finetuned.pt` exists. If the repository changes its model name or Hugging Face assets, update the model path based on its current README.
- Stockfish move acceptance is based on quick engine evaluations. It is a training heuristic, not proof that every accepted move solves the puzzle's intended tactic.
- Render's free service has resource and availability constraints. CPU-only inference over a large scanned book can take time; split very large books if extraction hits limits.

## No database

The application intentionally removes SQLite, user registration/login, database-backed positions, progress history, and persistent server-side saves from the supplied trainer. No database service is needed.
