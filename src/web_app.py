from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.app import service


DEFAULT_CHECKPOINT_PATH = Path(
    "checkpoints/best-epoch=18-val_ndcg_small=0.0239.ckpt"
)

DEFAULT_CONFIG_PATH = Path(
    "configs/model_params/gcn_L4_emb64_comp32_do0p1_lr3e-4_acc4_single_gpu_grad_accum_checkpointing.yaml"
)

DEFAULT_QUANTIZE_INT8 = False
DEFAULT_ITEM_BLOCK_SIZE = 1024


def load_default_model() -> dict[str, Any]:
    """
    Loads the fixed demo checkpoint once.
    The user does not need to know anything about checkpoints/configs.
    """
    if service.state.loaded:
        return {"status": "already_loaded"}

    missing_files = [
        str(path)
        for path in [DEFAULT_CHECKPOINT_PATH, DEFAULT_CONFIG_PATH]
        if not path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Cannot start demo app because required files are missing: "
            + ", ".join(missing_files)
        )

    return service.load(
        checkpoint_path=str(DEFAULT_CHECKPOINT_PATH),
        config_path=str(DEFAULT_CONFIG_PATH),
        quantize_int8=DEFAULT_QUANTIZE_INT8,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_default_model()
    yield


app = FastAPI(
    title="Dynamic GNN Recommender Demo",
    version="0.1.0",
    lifespan=lifespan,
)


class UIRecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    k: int = Field(default=5, ge=1, le=50)
    branch: str = Field(default="small")


class UIRecommendFromPreferencesRequest(BaseModel):
    gender: str
    age: int = Field(..., ge=1)
    occupation: str
    selected_movie_ids: list[int] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=50)
    item_block_size: int = Field(default=DEFAULT_ITEM_BLOCK_SIZE, ge=1)
    branch: str = Field(default="small")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": service.state.loaded,
        "num_items": service.state.num_items,
    }


@app.post("/api/recommend")
def api_recommend(payload: UIRecommendRequest) -> dict[str, Any]:
    if not service.state.loaded:
        raise HTTPException(status_code=500, detail="Model was not loaded.")

    return service.recommend(
        user_id=payload.user_id,
        k=payload.k,
        item_block_size=DEFAULT_ITEM_BLOCK_SIZE,
        branch=payload.branch,
    )


@app.get("/api/movies_onboarding")
def api_movies_onboarding(n: int = 80) -> dict[str, Any]:
    return service.movies_onboarding(n=n)


@app.post("/api/recommend_from_preferences")
def api_recommend_from_preferences(payload: UIRecommendFromPreferencesRequest) -> dict[str, Any]:
    return service.recommend_from_preferences(payload)

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Dynamic GNN Recommender</title>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f4f6fb;
            color: #111827;
        }

        .page {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px;
        }

        .card {
            width: 100%;
            max-width: 980px;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.12);
            padding: 32px;
        }

        h1 {
            margin: 0 0 8px;
            font-size: 32px;
        }

        .subtitle {
            margin: 0 0 28px;
            color: #6b7280;
            line-height: 1.5;
        }

        .form {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
            gap: 16px;
            align-items: start;
            margin-bottom: 24px;
        }

        label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #374151;
        }

        input {
            width: 100%;
            box-sizing: border-box;
            border: 1px solid #d1d5db;
            border-radius: 14px;
            padding: 12px 14px;
            font-size: 16px;
            outline: none;
        }

        input:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }

        button {
            border: none;
            border-radius: 14px;
            padding: 13px 22px;
            font-size: 16px;
            font-weight: 700;
            background: #2563eb;
            color: white;
            cursor: pointer;
            white-space: nowrap;
            margin-top: 22px;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            background: #93c5fd;
            cursor: default;
        }

        .field-hint {
            margin-top: 6px;
            font-size: 13px;
            color: #6b7280;
        }

        .status {
            margin: 12px 0 0;
            color: #6b7280;
        }

        .error {
            margin-top: 16px;
            padding: 14px 16px;
            border-radius: 14px;
            background: #fee2e2;
            color: #991b1b;
            display: none;
        }

        .results {
            width: 100%;
            margin-top: 24px;
            display: none;
        }

        .user-card {
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 16px 18px;
            background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 100%);
            margin-bottom: 18px;
        }

        .user-card h3 {
            margin: 0 0 4px;
            font-size: 18px;
            color: #1e3a8a;
        }

        .user-subtitle {
            margin: 0 0 12px;
            color: #6b7280;
            font-size: 13px;
        }

        .user-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px 14px;
        }

        .user-pill {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 10px 12px;
        }

        .user-field {
            display: block;
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 4px;
        }

        .user-value {
            font-size: 15px;
            font-weight: 700;
            color: #111827;
        }

        table {
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 16px;
        }

        th, td {
            text-align: left;
            padding: 14px 16px;
            border-bottom: 1px solid #e5e7eb;
            vertical-align: middle;
        }

        th {
            background: #f9fafb;
            color: #374151;
            font-size: 14px;
        }

        tr:last-child td {
            border-bottom: none;
        }

        th:nth-child(1), td:nth-child(1) {
            width: 70px;
        }

        th:nth-child(2), td:nth-child(2) {
            width: 45%;
        }

        th:nth-child(3), td:nth-child(3) {
            width: 30%;
        }

        th:nth-child(4), td:nth-child(4) {
            width: 110px;
        }

        .rank {
            font-weight: 700;
            color: #2563eb;
        }

        .movie-title {
            font-weight: 700;
            line-height: 1.25;
        }

        .movie-meta {
            margin-top: 4px;
            font-size: 13px;
            color: #6b7280;
        }

        .genres-cell {
            color: #374151;
        }

        .score-cell {
            font-variant-numeric: tabular-nums;
        }

        .onboarding-movie-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            font-size: 14px;
            color: #111827;
            cursor: pointer;
        }

        .onboarding-movie-row input[type="checkbox"] {
            width: auto;
            margin: 0;
            padding: 0;
            transform: translateY(1px);
        }

        .note {
            margin-top: 18px;
            color: #6b7280;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .branches-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 20px;
            align-items: start;
        }
        
        .branch-card {
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px;
            background: #ffffff;
        }
        
        .branch-header h3 {
            margin: 0;
            font-size: 20px;
        }
        
        .branch-header p {
            margin: 6px 0 16px;
            color: #6b7280;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .branch-card table {
            font-size: 14px;
        }
        
        .branch-card th,
        .branch-card td {
            padding: 12px 10px;
        }
        
        .branch-card th:nth-child(1),
        .branch-card td:nth-child(1) {
            width: 55px;
        }
        
        .branch-card th:nth-child(2),
        .branch-card td:nth-child(2) {
            width: 42%;
        }
        
        .branch-card th:nth-child(3),
        .branch-card td:nth-child(3) {
            width: 30%;
        }
        
        .branch-card th:nth-child(4),
        .branch-card td:nth-child(4) {
            width: 80px;
        }

        @media (max-width: 700px) {
            .form {
                grid-template-columns: 1fr;
            }

            button {
                width: 100%;
                margin-top: 0;
            }

            table {
                table-layout: auto;
            }
            
            .branches-grid {
                grid-template-columns: 1fr;
            }

            .user-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="card">
            <h1>Dynamic GNN Recommender</h1>
            <p class="subtitle">
                Enter a user ID and choose how many recommendations to show.
                The model is already loaded in the background.
            </p>

            <div class="form">
                <div>
                    <label for="userId">User ID</label>
                    <input id="userId" type="text" value="6" />
                    <div class="field-hint">Enter an integer from 0 to 942.</div>
                </div>

                <div>
                    <label for="k">Number of recommendations</label>
                    <input id="k" type="text" value="5" />
                    <div class="field-hint">Enter an integer from 1 to 50.</div>
                </div>

                <button id="recommendBtn" onclick="recommend()">Recommend</button>
            </div>

            <hr style="margin: 20px 0; border: none; border-top: 1px solid #e5e7eb;" />
            <h2 style="margin: 0 0 8px;">Cold-start recommendations</h2>
            <p class="subtitle" style="margin-bottom: 12px;">Fill profile and select watched movies from onboarding list.</p>
            <div class="form" style="grid-template-columns: repeat(4,minmax(0,1fr));">
                <div><label for="prefGender">Gender</label><input id="prefGender" type="text" value="M" /></div>
                <div><label for="prefAge">Age</label><input id="prefAge" type="text" value="25" /></div>
                <div><label for="prefOccupation">Occupation</label><input id="prefOccupation" type="text" value="student" /></div>
                <div><label for="prefK">k</label><input id="prefK" type="text" value="5" /></div>
            </div>
            <button id="loadOnboardingBtn" onclick="loadOnboardingMovies()" style="margin-top:0;">Load onboarding movies</button>
            <button id="recommendFromPrefsBtn" onclick="recommendFromPreferences()" style="margin-top:0; margin-left:8px; display:none;">Recommend from preferences</button>
            <div id="onboardingMovies" style="margin-top:12px; max-height:240px; overflow:auto; border:1px solid #e5e7eb; border-radius:12px; padding:10px;"></div>

            <p id="status" class="status">Ready.</p>
            <div id="error" class="error"></div>

            <div id="results" class="results">
                <h2>Recommendations</h2>
            
                <section id="userInfo" class="user-card" style="display: none;">
                    <h3>User profile</h3>
                    <p class="user-subtitle">Demographic metadata for the selected user.</p>
                    <div class="user-grid">
                        <div class="user-pill"><span class="user-field">Age</span><span class="user-value" id="userAge">—</span></div>
                        <div class="user-pill"><span class="user-field">Gender</span><span class="user-value" id="userGender">—</span></div>
                        <div class="user-pill"><span class="user-field">Occupation</span><span class="user-value" id="userOccupation">—</span></div>
                    </div>
                </section>

                <div class="branches-grid">
                    <section class="branch-card">
                        <div class="branch-header">
                            <h3>Small branch</h3>
                            <p>Compressed embeddings from the same checkpoint.</p>
                        </div>
            
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Movie</th>
                                    <th>Genres</th>
                                    <th>Score</th>
                                </tr>
                            </thead>
                            <tbody id="smallResultsBody"></tbody>
                        </table>
                    </section>
            
                    <section class="branch-card">
                        <div class="branch-header">
                            <h3>Big branch</h3>
                            <p>Full embeddings from the same checkpoint.</p>
                        </div>
            
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Movie</th>
                                    <th>Genres</th>
                                    <th>Score</th>
                                </tr>
                            </thead>
                            <tbody id="bigResultsBody"></tbody>
                        </table>
                    </section>
                </div>
            
                <p class="note">
                    Score is the model relevance score: the higher it is, the higher the movie is ranked for this user.
                </p>
            </div>
        </section>
    </main>

    <script>
        function parseBoundedInteger(value, fieldName, minValue, maxValue) {
            const text = String(value).trim();

            if (!/^[0-9]+$/.test(text)) {
                throw new Error(
                    `Incorrect input data: ${fieldName} must be an integer from ${minValue} to ${maxValue}.`
                );
            }

            const parsed = Number(text);

            if (!Number.isInteger(parsed) || parsed < minValue || parsed > maxValue) {
                throw new Error(
                    `Incorrect input data: ${fieldName} must be an integer from ${minValue} to ${maxValue}.`
                );
            }

            return parsed;
        }

        function formatBackendError(data) {
            if (!data) {
                return "Incorrect input data.";
            }

            if (typeof data.detail === "string") {
                if (data.detail.toLowerCase().includes("user")) {
                    return "Incorrect input data: user_id must be an integer from 0 to 942.";
                }

                if (data.detail.toLowerCase().includes("k")) {
                    return "Incorrect input data: k must be an integer from 1 to 50.";
                }

                return data.detail;
            }

            return "Incorrect input data.";
        }

        async function fetchRecommendations(userId, k, branch) {
            const response = await fetch("/api/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    user_id: userId,
                    k: k,
                    branch: branch
                })
            });
        
            const data = await response.json();
        
            if (!response.ok) {
                throw new Error(formatBackendError(data));
            }
        
            return data;
        }
        
        function renderUserInfo(data) {
            const info = data.user || {};
            document.getElementById("userAge").textContent = info.age ?? "Unknown";
            document.getElementById("userGender").textContent = info.gender ?? "Unknown";
            document.getElementById("userOccupation").textContent = info.occupation ?? "Unknown";
            document.getElementById("userInfo").style.display = "block";
        }

        function renderRecommendations(data, tbodyId) {
            const resultsBody = document.getElementById(tbodyId);
            resultsBody.innerHTML = "";
        
            data.recommendations.forEach((rec, index) => {
                const row = document.createElement("tr");
        
                const title = rec.title || `MovieLens item ${rec.raw_item_id || rec.item_id}`;
                const genres = rec.genres && rec.genres.length > 0
                    ? rec.genres.join(", ")
                    : "Unknown";
        
                row.innerHTML = `
                    <td class="rank">#${index + 1}</td>
                    <td>
                        <div class="movie-title">${title}</div>
                        <div class="movie-meta">MovieLens ID: ${rec.movie_id || "unknown"}</div>
                    </td>
                    <td class="genres-cell">${genres}</td>
                    <td class="score-cell">${rec.score.toFixed(4)}</td>
                `;
        
                resultsBody.appendChild(row);
            });
        }

        async function loadOnboardingMovies() {
            const response = await fetch('/api/movies_onboarding?n=80');
            const data = await response.json();
            if (!response.ok) throw new Error(formatBackendError(data));
            const container = document.getElementById('onboardingMovies');
            const recommendBtn = document.getElementById('recommendFromPrefsBtn');
            container.innerHTML = '';
            recommendBtn.style.display = 'none';
            data.movies.forEach((movie) => {
                const row = document.createElement('label');
                row.className = 'onboarding-movie-row';
                row.innerHTML = `<input type="checkbox" class="onboarding-movie" value="${movie.item_id}" /> ${movie.title}`;
                row.querySelector('input').addEventListener('change', () => {
                    const selectedCount = document.querySelectorAll('.onboarding-movie:checked').length;
                    recommendBtn.style.display = selectedCount > 0 ? 'inline-block' : 'none';
                });
                container.appendChild(row);
            });
            document.getElementById('status').textContent = `Loaded ${data.movies.length} onboarding movies.`;
        }

        async function recommendFromPreferences() {
            const age = parseBoundedInteger(document.getElementById('prefAge').value, 'age', 1, 120);
            const k = parseBoundedInteger(document.getElementById('prefK').value, 'k', 1, 50);
            const gender = document.getElementById('prefGender').value.trim();
            const occupation = document.getElementById('prefOccupation').value.trim();
            const selected = Array.from(document.querySelectorAll('.onboarding-movie:checked')).map((x) => Number(x.value));
            if (selected.length === 0) throw new Error('Please select at least one movie.');

            const requestPayload = {
                gender,
                age,
                occupation,
                selected_movie_ids: selected,
                k,
                item_block_size: 1024,
            };

            const [smallResponse, bigResponse] = await Promise.all([
                fetch('/api/recommend_from_preferences', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({...requestPayload, branch: 'small'})
                }),
                fetch('/api/recommend_from_preferences', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({...requestPayload, branch: 'big'})
                })
            ]);

            const smallData = await smallResponse.json();
            const bigData = await bigResponse.json();
            if (!smallResponse.ok) throw new Error(formatBackendError(smallData));
            if (!bigResponse.ok) throw new Error(formatBackendError(bigData));

            renderUserInfo(smallData);
            renderRecommendations(smallData, 'smallResultsBody');
            renderRecommendations(bigData, 'bigResultsBody');
            document.getElementById('results').style.display = 'block';
            document.getElementById('status').textContent = 'Cold-start recommendations ready for both branches.';
        }

        async function recommend() {
            const userIdText = document.getElementById("userId").value;
            const kText = document.getElementById("k").value;
        
            const button = document.getElementById("recommendBtn");
            const status = document.getElementById("status");
            const errorBox = document.getElementById("error");
            const results = document.getElementById("results");
        
            errorBox.style.display = "none";
            results.style.display = "none";
            document.getElementById("smallResultsBody").innerHTML = "";
            document.getElementById("bigResultsBody").innerHTML = "";
            document.getElementById("userInfo").style.display = "none";
        
            try {
                const userId = parseBoundedInteger(userIdText, "user_id", 0, 942);
                const k = parseBoundedInteger(kText, "k", 1, 50);
        
                button.disabled = true;
                status.textContent = "Computing recommendations for both branches...";
        
                const [smallData, bigData] = await Promise.all([
                    fetchRecommendations(userId, k, "small"),
                    fetchRecommendations(userId, k, "big")
                ]);
        
                renderUserInfo(smallData);
                renderRecommendations(smallData, "smallResultsBody");
                renderRecommendations(bigData, "bigResultsBody");
        
                results.style.display = "block";
                status.textContent = `Recommendations for user ${userId}.`;
            } catch (error) {
                errorBox.textContent = error.message || "Incorrect input data.";
                errorBox.style.display = "block";
                status.textContent = "Something went wrong.";
            } finally {
                button.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run user-friendly recommender demo")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("src.web_app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
