# DocuMind Deployment Guide

End-to-end deployment of DocuMind to Azure. Backend on App Service (Linux, Python 3.12), frontend on Static Web Apps.

This guide documents what actually worked after iterative debugging. The traps it warns about are real ones we hit — don't skip them.

---

## Architecture

```
┌─────────────────────────┐         ┌────────────────────────────┐
│  documind-web (SWA)     │ ──────▶ │  documind-api (App Service)│
│  React + Vite           │  HTTPS  │  FastAPI + Gunicorn        │
│  Static Web Apps        │         │  Python 3.12 / Linux B1    │
└─────────────────────────┘         └──────────┬─────────────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                  ┌───────▼─────┐  ┌──────────▼────────┐  ┌────────▼────────┐
                  │ Azure       │  │ Azure AI Search   │  │ Azure Document  │
                  │ OpenAI      │  │ (vector + BM25)   │  │ Intelligence    │
                  └─────────────┘  └───────────────────┘  └─────────────────┘
```

Resources (resource group `documind-rg`, region Central India):
- `documind-api` — App Service, B1 plan
- `documind-web` — Static Web App, free tier
- `documind-openai` — Azure OpenAI (gpt-4o-mini + text-embedding-3-small)
- `documind-search2` — Azure AI Search (basic)
- `documind-doc-ai` — Document Intelligence

---

## Pre-flight checklist

Before starting any deploy, confirm all of these:

- [ ] You're in the right Azure subscription (`az account show` shows the right ID)
- [ ] Local backend works (`uvicorn DocuMind.api.main:app --reload` runs, `/api/health` returns 200)
- [ ] Local frontend works (`npm run dev`, register/login/upload/ask all work end-to-end)
- [ ] `.venv312` is activated, not `.venv` (Python 3.12, not 3.14)
- [ ] All Azure resources exist and are reachable from your network
- [ ] `deploy.zip`, `logs.zip`, `logs_extracted/` from previous attempts are deleted

```powershell
cd C:\mywork\DocuMind.Python
.\.venv312\Scripts\Activate.ps1
Remove-Item deploy.zip, logs.zip -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force logs_extracted -ErrorAction SilentlyContinue
az account show --query "{name:name, id:id}" -o table
```

---

## Phase 1 — Wipe stale data

Old test users and corrupted indexes will cause silent issues. Wipe both before deploying.

### 1.1 Wipe the live SQLite database

Open Kudu: **https://documind-api.scm.azurewebsites.net**

In the right column, click **SSH (App)** (the link with the external arrow icon). A new tab opens with a bash terminal inside the running app container.

In that terminal, run:

```bash
ls -la /home/site/wwwroot/documind.db 2>/dev/null && rm /home/site/wwwroot/documind.db && echo "Deleted" || echo "No file at that path"
```

If the response is "No file at that path", also check whether old startup scripts wrote it elsewhere:

```bash
find / -name "documind.db" 2>/dev/null
```

Delete any paths found with `rm <full-path>`.

> **Note:** Older guides reference "Debug console → Bash" — that menu was removed in a Kudu UI refresh. Use **SSH (App)** instead. **SSH (Kudu)** also works (different container, same `/home/site` filesystem view).

### 1.2 Wipe the Azure Search index

Run locally:

```powershell
python wipe_search_index.py
```

Type `yes` at the prompt. The script auto-discovers the key field, batch-deletes all chunks via the SDK.

---

## Phase 2 — Backend deploy

### 2.1 Verify required environment variables on App Service

This command lists every setting with sensitive keys redacted to first 8 chars so you can spot them but not expose them:

```powershell
az webapp config appsettings list `
  --resource-group documind-rg `
  --name documind-api `
  --query "[].{name:name, value:value}" `
  -o tsv | ForEach-Object {
    $parts = $_ -split "`t", 2
    $name = $parts[0]
    $val = $parts[1]
    if ($name -match "KEY|SECRET|TOKEN|PASSWORD") {
      $val = $val.Substring(0, [Math]::Min(8, $val.Length)) + "...[redacted]"
    }
    "{0,-50} = {1}" -f $name, $val
  }
```

You MUST see all of these (with non-empty values):

| Setting | Required value or pattern |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | `https://documind-openai.openai.azure.com/` |
| `AZURE_OPENAI_KEY` | (any non-empty value) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o-mini` (or actual deployment name) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` (or actual deployment name) |
| `AZURE_SEARCH_ENDPOINT` | `https://documind-search2.search.windows.net` |
| `AZURE_SEARCH_KEY` | (any non-empty value) |
| `AZURE_SEARCH_INDEX_NAME` | `documind-pdf-generic5-index` |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | `https://documind-doc-ai.cognitiveservices.azure.com/` |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | (any non-empty value) |
| `SECRET_KEY` | (any non-empty value, 32+ chars) |
| `LLM_PROVIDER` | exactly `azure` |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | exactly `true` (lowercase string) |

Optional but recommended (LangSmith tracing):
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<your-key>`
- `LANGCHAIN_PROJECT=DocuMind`

**Common gotchas:**
- `*_DEPLOYMENT` settings are NOT model names — they are the *deployment name* you chose in Azure OpenAI Studio. May be the same string as the model, may differ.
- `LLM_PROVIDER` must be lowercase `azure`, not `Azure` or `AZURE`.
- `SCM_DO_BUILD_DURING_DEPLOYMENT` must be lowercase string `true`. The string `True`, the boolean true, or `1` all behave inconsistently — verify by running this:
  ```powershell
  az webapp config appsettings list `
    --resource-group documind-rg `
    --name documind-api `
    --query "[?name=='SCM_DO_BUILD_DURING_DEPLOYMENT']" -o json
  ```
  The `value` field MUST be `"true"` (string with quotes in JSON output). If it's `null` or boolean `true` (no quotes), set it again.

If anything is missing, set it (replace `<value>`):

```powershell
az webapp config appsettings set `
  --resource-group documind-rg `
  --name documind-api `
  --settings KEY_NAME=<value>
```

Multiple at once:

```powershell
az webapp config appsettings set `
  --resource-group documind-rg `
  --name documind-api `
  --settings AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 2.2 Delete `PYTHONPATH` if present

This is critical. Old startup scripts often left `PYTHONPATH` as an App Service-level setting pointing at stale `/tmp/<old-oryx-id>` paths. The new `startup.sh` sets `PYTHONPATH` at runtime to the correct value — an App Service-level setting overrides this and breaks imports.

```powershell
az webapp config appsettings list `
  --resource-group documind-rg `
  --name documind-api `
  --query "[?name=='PYTHONPATH']" -o tsv
```

If anything returns, delete it:

```powershell
az webapp config appsettings delete `
  --resource-group documind-rg `
  --name documind-api `
  --setting-names PYTHONPATH
```

### 2.3 Verify CORS is empty at Azure level

CORS is handled by FastAPI middleware. Azure-level CORS would interfere — confirm it's empty:

```powershell
az webapp cors show `
  --resource-group documind-rg `
  --name documind-api
```

Should return `{"allowedOrigins": [], "supportCredentials": false}`. If anything's there, remove it:

```powershell
az webapp cors remove `
  --resource-group documind-rg `
  --name documind-api `
  --allowed-origins *
```

### 2.4 Confirm startup file setting

```powershell
az webapp config show `
  --resource-group documind-rg `
  --name documind-api `
  --query "appCommandLine" -o tsv
```

Must return: `startup.sh`

If empty or different:

```powershell
az webapp config set `
  --resource-group documind-rg `
  --name documind-api `
  --startup-file "startup.sh"
```

### 2.5 Verify startup.sh is correct (LF line endings, no BOM)

The most common silent killer of Azure Linux Python deploys is Windows line endings or BOM in `startup.sh`. Run this to ensure both:

```powershell
cd C:\mywork\DocuMind.Python

# UTF-8 without BOM, LF line endings
$content = (Get-Content startup.sh -Raw) -replace "`r`n","`n"
[System.IO.File]::WriteAllText(
  "$PWD\startup.sh",
  $content,
  (New-Object System.Text.UTF8Encoding $false)
)

# Verify
$bytes = [System.IO.File]::ReadAllBytes("$PWD\startup.sh")
if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    Write-Host "WARNING: startup.sh has UTF-8 BOM" -ForegroundColor Red
} else {
    Write-Host "startup.sh: no BOM" -ForegroundColor Green
}
$crlfCount = ([Text.Encoding]::UTF8.GetString($bytes) -split "`r`n").Count - 1
if ($crlfCount -gt 0) {
    Write-Host "WARNING: $crlfCount CRLF line endings found" -ForegroundColor Red
} else {
    Write-Host "startup.sh: LF line endings" -ForegroundColor Green
}
```

Both must say "no BOM" and "LF line endings" before continuing.

> **Why this matters:** Bash on Linux interprets the BOM as part of the shebang and produces error `: not found` even though the script eventually runs. CRLF causes errors like `\r: command not found`. Either can produce confusing failures hours later.

### 2.6 Build deploy.zip

```powershell
Remove-Item deploy.zip -ErrorAction SilentlyContinue

python -c "
import zipfile, os
exclude_dirs = {'.venv', '.venv312', '.git', '__pycache__', 'antenv', '.ruff_cache', 'node_modules', 'tests', '.github', '.vscode', 'logs_extracted'}
exclude_files = {'documind.db', 'gmail_credentials.json', 'gmail_token.json', '.env', 'deploy.zip', 'wipe_search_index.py', 'migrate.py', 'logs.zip'}
include = ['DocuMind', 'requirements.txt', 'runtime.txt', 'pyproject.toml', 'startup.sh']
with zipfile.ZipFile('deploy.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for item in include:
        if os.path.isfile(item):
            zf.write(item)
        elif os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for file in files:
                    if file in exclude_files or file.endswith('.pyc'):
                        continue
                    filepath = os.path.join(root, file)
                    zf.write(filepath, filepath.replace(os.sep, '/'))
print('Done!')
"
```

Verify the zip looks right:

```powershell
python -c "import zipfile; z=zipfile.ZipFile('deploy.zip'); names=z.namelist(); print(f'{len(names)} files'); [print(n) for n in names if '/' not in n]"
```

Expected output — these files at the root, plus a `DocuMind/` tree:
```
pyproject.toml
requirements.txt
runtime.txt
startup.sh
```

### 2.7 Open log tail in a separate window

In a **new** PowerShell window (don't close your main one):

```powershell
az webapp log tail --resource-group documind-rg --name documind-api
```

This streams the boot log live. Keep it visible while the deploy runs.

> **Note:** `az webapp log tail` is unreliable — sometimes drops connections. If output stops mid-deploy, that doesn't mean the deploy failed. Trust `az webapp log download` (Phase 5) over the live tail.

### 2.8 Deploy

Back in your main window:

```powershell
az webapp deploy `
  --resource-group documind-rg `
  --name documind-api `
  --src-path deploy.zip `
  --type zip `
  --async true
```

**Why `--async true`:** sync mode holds one HTTP connection open for the entire build. Azure's gateway times out at ~230 seconds, but pip-installing your requirements (numpy, scikit-learn, scipy, opencv, PyMuPDF, etc.) takes 5–10 minutes on B1. Sync mode fails with a 504, even though Azure builds successfully. Async submits the job and returns immediately.

The CLI output will be brief — that's expected. Watch the log tail window.

**Total time:** ~7 minutes for the full build + boot.

### 2.9 Verify boot

Wait 7 minutes after deploy submitted, then:

```powershell
curl https://documind-api.azurewebsites.net/api/health
```

Expected: `{"status":"ok"}`

If you get 503: wait another 60 seconds, retry once. Cold-start can take a moment after build completes.

If still failing, jump to Phase 5 (Diagnostics) — don't redeploy blindly.

### 2.10 Sanity check auth + DB + Pydantic

```powershell
$body = @{
    username = "test_$(Get-Random)"
    email = "test@example.com"
    password = "TestPass123!"
    company = "TestCo"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://documind-api.azurewebsites.net/api/auth/register" `
  -Method POST `
  -Body $body `
  -ContentType "application/json"
```

Expected: returns a JSON object with `token`, `username`, `email`, `company` fields. If you get this, the backend is fully functional — DB writes work, auth works, Pydantic validation works.

---

## Phase 3 — Frontend deploy

### 3.1 Prerequisites at the frontend repo root

`C:\mywork\DocuMind.Web\` should contain these two files at the root level (same as `package.json`):

**`.env.production`** — baked into the build at compile time, must include:
```
VITE_API_URL=https://documind-api.azurewebsites.net
```

Create with PowerShell (Windows hides dot-files in Explorer):
```powershell
cd C:\mywork\DocuMind.Web
Set-Content -Path .env.production -Value "VITE_API_URL=https://documind-api.azurewebsites.net"

# Verify the filename is exactly .env.production (NOT .env.production.txt)
Get-ChildItem .env.production | Select-Object Name
Get-Content .env.production
```

**`staticwebapp.config.json`** — SPA fallback routing + security headers. Without this, deep links like `/login` 404 on SWA.

### 3.2 Confirm Node version

Vite 8 requires Node ≥20.19. Check yours:

```powershell
node --version
```

If lower, upgrade via [nodejs.org](https://nodejs.org) before continuing.

### 3.3 Build

```powershell
cd C:\mywork\DocuMind.Web
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
npm run build
```

Should finish in ~30 seconds with no errors. Output goes to `dist/`.

### 3.4 Verify the API URL was baked in correctly

This is critical. Vite bakes env vars at build time, not runtime. If `.env.production` was missing or malformed, the build will silently default to localhost.

```powershell
# MUST find a match
Select-String -Path dist/assets/*.js -Pattern "documind-api.azurewebsites" | Select-Object -First 1

# MUST find NOTHING
Select-String -Path dist/assets/*.js -Pattern "localhost:8000"
```

If `documind-api` doesn't appear OR `localhost:8000` does — STOP. Don't deploy. Re-check `.env.production` exists at the right place with the right content, then rebuild.

### 3.5 Get SWA deployment token

```powershell
az staticwebapp secrets list `
  --name documind-web `
  --resource-group documind-rg `
  --query "properties.apiKey" -o tsv
```

Copy the output (long token). You'll need it next.

### 3.6 Deploy to SWA production slot

```powershell
swa deploy ./dist --env production --deployment-token <PASTE_TOKEN>
```

**Important:** `--env production` matters. Without it, the CLI deploys to the preview slot, and your custom domain points to production. Skipping this flag means "deploy succeeded" but the live site still shows the old build.

Wait for "Deployment complete" — usually 30–60 seconds.

### 3.7 Verify

Open **https://documind-web.azurestaticapps.net** (or your custom domain). Should load the React app.

Open browser DevTools → Network tab → reload page. The XHR requests to `/api/...` should hit `https://documind-api.azurewebsites.net`, not localhost.

---

## Phase 4 — Smoke test

End-to-end test through the live UI. All of these should work:

- [ ] Page loads at production URL
- [ ] Theme toggle works, persists across reload
- [ ] Register a new account with company field
- [ ] Sign out, sign in
- [ ] Company field shows in top-right header (DM Serif Display, 20px)
- [ ] Upload a small PDF — succeeds, appears in sidebar under chosen category
- [ ] Ask a question — get prose answer + citation badges
- [ ] Click a citation badge — expands to show page + doc + verbatim quote
- [ ] Copy button on assistant message — shows "✓ Copied" 1.5s
- [ ] Trash icon visible on doc rows at idle (opacity 0.7)
- [ ] Hover trash icon — turns red with tinted background
- [ ] Click trash — confirm dialog (smart message if last in category)
- [ ] After confirm, doc disappears optimistically; reload still shows gone

If anything fails, check:
1. Browser DevTools console for errors
2. Network tab for failing XHR (4xx/5xx)
3. Azure log tail for backend exceptions

---

## Phase 5 — Diagnostics (when something breaks)

When the live tail goes silent or you're not sure what's running, download the actual logs:

```powershell
cd C:\mywork\DocuMind.Python
Remove-Item logs.zip, logs_extracted -Recurse -Force -ErrorAction SilentlyContinue

az webapp log download `
  --resource-group documind-rg `
  --name documind-api `
  --log-file logs.zip

Expand-Archive logs.zip -DestinationPath logs_extracted -Force

# Most recent docker.log has the boot trace
$latest = Get-ChildItem logs_extracted -Recurse -Filter "*docker*.log" |
          Sort-Object LastWriteTime -Descending |
          Select-Object -First 1
Write-Host "Latest log: $($latest.FullName)" -ForegroundColor Cyan
Get-Content $latest.FullName -Tail 200
```

What to look for, in order:

1. **`[startup] App import OK`** — if present, Python imports succeeded
2. **`[INFO] Starting gunicorn`** — gunicorn launched
3. **`[INFO] Booting worker with pid: ...`** — workers spawning
4. **`[INFO] Listening at: http://0.0.0.0:8000`** — ready for requests

Failure signatures:

| Symptom in log | Cause | Fix |
|---|---|---|
| `: not found` after shebang line | UTF-8 BOM in startup.sh | Phase 2.5 |
| `\r: command not found` | CRLF line endings | Phase 2.5 |
| `[startup] FATAL: APP_PATH not set...` | Oryx didn't run | Check `SCM_DO_BUILD_DURING_DEPLOYMENT=true` |
| `[startup] FATAL: DocuMind/ folder not found` | deploy.zip is missing source | Re-check Phase 2.6 zip contents |
| `ModuleNotFoundError: No module named 'X'` (where X is in requirements.txt) | pip install failed silently | Check Oryx build log via deployments URL |
| `ModuleNotFoundError: No module named 'DocuMind'` | startup.sh used wrong cwd | Should be fixed by current startup.sh — verify deploy actually picked up the new file |
| `Worker failed to boot` with no stack | Pre-validation didn't run | startup.sh from this guide always pre-validates — old version still in zip? |
| `gunicorn.errors.HaltServer` exit code 127 | gunicorn not on PATH | startup.sh should add `$ANTENV/bin` to PATH — verify deploy picked up new file |
| Container repeatedly restarts, exit code 3 | Worker boot exceptions | Look earlier in log for the actual Python traceback before "Worker exited" |

### Force a fresh boot to capture clean logs

If logs from previous failed boots are mixed with current ones:

```powershell
az webapp restart --resource-group documind-rg --name documind-api
Start-Sleep -Seconds 60
# Then run the log download command above
```

### Check deployment status if `az webapp deploy` 504'd

```powershell
$creds = az webapp deployment list-publishing-credentials `
  --resource-group documind-rg --name documind-api `
  --query "{user:publishingUserName, pwd:publishingPassword}" -o json | ConvertFrom-Json
$pair = "$($creds.user):$($creds.pwd)"
$auth = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
Invoke-RestMethod `
  -Uri "https://documind-api.scm.azurewebsites.net/api/deployments/latest" `
  -Headers @{Authorization=$auth} |
  Select-Object status, status_text, progress, complete, end_time
```

Status: `0`=Pending, `1`=InProgress, `3`=Failed, `4`=Success, `5`=Building. The CLI 504 doesn't mean Azure failed — only retry deploy if status shows `3`.

---

## Phase 6 — Rollback

If a deploy goes wrong and you need to revert:

```powershell
# List recent deployments (most recent first)
az webapp deployment list `
  --resource-group documind-rg `
  --name documind-api `
  --query "[].{id:id, status:status, message:message, time:received_time}" -o table
```

Find the last working deployment ID (status `4`), then redeploy that build. Azure doesn't support direct redeploy of a previous artifact via CLI — easiest is to keep your last-known-good `deploy.zip` archived and redeploy that file.

**Recommendation:** before any deploy, copy `deploy.zip` to `deploy.zip.backup`. If new deploy breaks, just redeploy the backup.

---

## Known issues / future work

These aren't blockers but are worth tracking:

### SQLite is on ephemeral storage

The DocuMind code lives at `/tmp/<oryx-hash>/` at runtime. SQLite writes `documind.db` to the same location. **`/tmp` is wiped when the container restarts** (Azure rotates instances, every redeploy). Users you register WILL disappear on the next restart.

**Fix:** modify `database.py` to use `/home/site/documind.db` (persistent across restarts), or set `DB_PATH` env var. Two ways:

Option A — environment-driven (preferred):
```python
# In database.py
import os
DB_PATH = os.environ.get("DB_PATH", "documind.db")
```
Then set `DB_PATH=/home/site/documind.db` as an App Service env var.

Option B — symlink at startup:
```bash
# In startup.sh, before launching gunicorn:
mkdir -p /home/site
[ -f /home/site/documind.db ] || touch /home/site/documind.db
ln -sf /home/site/documind.db "$APP_DIR/documind.db"
```

### Secret rotation needed

Any keys exposed in chat history during the deploy session should be rotated:
- `AZURE_OPENAI_KEY`
- `AZURE_SEARCH_KEY`
- `AZURE_DOCUMENT_INTELLIGENCE_KEY`
- `LANGCHAIN_API_KEY`
- `SECRET_KEY` (rotate this last and only when no users are mid-session — rotating SECRET_KEY invalidates all JWTs)

After rotation:
```powershell
az webapp config appsettings set `
  --resource-group documind-rg `
  --name documind-api `
  --settings AZURE_OPENAI_KEY=<new-value>
# Repeat for each, then:
az webapp restart --resource-group documind-rg --name documind-api
```

### Log alerts

Configure in Azure Portal:
- Alert when `/api/health` returns non-200 for >2 minutes
- Alert when container exit code != 0
- Alert when Azure OpenAI quota >80%

### Dev environment isolation

Currently dev and prod share the same Search index (`documind-pdf-generic5-index`). Create `documind-pdf-generic5-index-dev` for local development to prevent cross-contamination.

---

## Quick reference — full deploy from scratch

If everything's already configured and you just need to push a code change:

```powershell
cd C:\mywork\DocuMind.Python
.\.venv312\Scripts\Activate.ps1

# Backup last good build
Copy-Item deploy.zip deploy.zip.backup -ErrorAction SilentlyContinue

# Fix line endings, build zip, deploy
$content = (Get-Content startup.sh -Raw) -replace "`r`n","`n"
[System.IO.File]::WriteAllText("$PWD\startup.sh", $content, (New-Object System.Text.UTF8Encoding $false))

Remove-Item deploy.zip -ErrorAction SilentlyContinue
python -c "
import zipfile, os
exclude_dirs = {'.venv', '.venv312', '.git', '__pycache__', 'antenv', '.ruff_cache', 'node_modules', 'tests', '.github', '.vscode', 'logs_extracted'}
exclude_files = {'documind.db', '.env', 'deploy.zip', 'deploy.zip.backup', 'wipe_search_index.py', 'migrate.py', 'logs.zip'}
include = ['DocuMind', 'requirements.txt', 'runtime.txt', 'pyproject.toml', 'startup.sh']
with zipfile.ZipFile('deploy.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for item in include:
        if os.path.isfile(item): zf.write(item)
        elif os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for file in files:
                    if file in exclude_files or file.endswith('.pyc'): continue
                    fp = os.path.join(root, file)
                    zf.write(fp, fp.replace(os.sep, '/'))
print('zip ready')
"

az webapp deploy `
  --resource-group documind-rg --name documind-api `
  --src-path deploy.zip --type zip --async true

# Wait, then verify
Start-Sleep -Seconds 420
curl https://documind-api.azurewebsites.net/api/health
```

For frontend changes only:

```powershell
cd C:\mywork\DocuMind.Web
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
npm run build
Select-String -Path dist/assets/*.js -Pattern "documind-api.azurewebsites" | Select -First 1
$token = az staticwebapp secrets list --name documind-web --resource-group documind-rg --query "properties.apiKey" -o tsv
swa deploy ./dist --env production --deployment-token $token
```

---

## Appendix — required files in repo

### Backend (`C:\mywork\DocuMind.Python\`)

```
DocuMind/                  # source
requirements.txt           # gunicorn==23.0.0 pinned, no starlette pin
runtime.txt                # python-3.12
pyproject.toml
startup.sh                 # LF line endings, no BOM, uses APP_PATH
```

### Frontend (`C:\mywork\DocuMind.Web\`)

```
src/
public/
package.json               # engines: node >=20.19.0, no homepage field
vite.config.js
.env.production            # VITE_API_URL=https://documind-api.azurewebsites.net
staticwebapp.config.json   # SPA fallback + security headers
index.html
```

### Helper scripts (not deployed, kept local)

```
wipe_search_index.py       # wipes Azure Search index
migrate.py                 # SQLite schema migrations (mostly historical)
```
