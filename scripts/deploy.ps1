# MetaAI ML Platform - Hugging Face Spaces Deployment Script

Write-Host "MetaAI Deployment to Hugging Face Spaces" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Delete .git folder
Write-Host "Step 1: Deleting old .git folder..." -ForegroundColor Yellow
$gitPath = Join-Path -Path (Get-Location) -ChildPath ".git"
if (Test-Path $gitPath) {
    Remove-Item -Path $gitPath -Recurse -Force
    Write-Host "Git folder deleted" -ForegroundColor Green
} else {
    Write-Host "No existing .git folder found" -ForegroundColor Green
}
Write-Host ""

# Step 2: Create fresh .gitignore
Write-Host "Step 2: Creating fresh .gitignore..." -ForegroundColor Yellow
$gitignoreContent = @"
.venv/
.venv312/
venv/
__pycache__/
*.pyc
*.pyo
mlruns/
models/
*.db
*.sqlite
data/*.csv
.env
*.exe
*.js.map
*.map
node_modules/
*.egg-info/
dist/
build/
memory/
database/
"@

$gitignorePath = Join-Path -Path (Get-Location) -ChildPath ".gitignore"
Set-Content -Path $gitignorePath -Value $gitignoreContent -Encoding UTF8
Write-Host ".gitignore created" -ForegroundColor Green
Write-Host ""

# Step 3: git init
Write-Host "Step 3: Initializing fresh git repository..." -ForegroundColor Yellow
& git init
Write-Host ""

# Step 4: git remote add
Write-Host "Step 4: Adding Hugging Face Spaces remote..." -ForegroundColor Yellow
& git remote add space https://huggingface.co/spaces/Sujithac07/metaai-ml-platform
Write-Host ""

# Step 5: git add .
Write-Host "Step 5: Staging all files..." -ForegroundColor Yellow
& git add .
Write-Host ""

# Step 6: git commit
Write-Host "Step 6: Creating commit..." -ForegroundColor Yellow
& git commit -m "MetaAI fresh deploy"
Write-Host ""

# Step 7: git push
Write-Host "Step 7: Pushing to Hugging Face Spaces..." -ForegroundColor Yellow
& git push space master --force
Write-Host ""

# Final message
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "DONE - Check huggingface.co/spaces/Sujithac07/metaai-ml-platform" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
