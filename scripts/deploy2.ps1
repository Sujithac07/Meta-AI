# MetaAI ML Platform - HuggingFace Spaces Deployment Script v2

Write-Host "MetaAI Deployment to Hugging Face Spaces" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check git status
Write-Host "Step 1: Checking files..." -ForegroundColor Yellow
Write-Host ""
& git status
Write-Host ""

# Step 2: Add files
Write-Host "Step 2: Adding files..." -ForegroundColor Yellow
Write-Host ""
& git add .
Write-Host "Files added" -ForegroundColor Green
Write-Host ""

# Step 3: Commit
Write-Host "Step 3: Committing..." -ForegroundColor Yellow
Write-Host ""
& git commit -m "MetaAI full deploy"
Write-Host ""

# Step 4: Push to main (with fallback to master)
Write-Host "Step 4: Pushing to HuggingFace..." -ForegroundColor Yellow
Write-Host ""

$pushSuccess = $false

# Try main first
Write-Host "Attempting to push to main..." -ForegroundColor Cyan
try {
    & git push space main --force 2>&1
    $pushSuccess = $true
    Write-Host "Successfully pushed to main" -ForegroundColor Green
} catch {
    Write-Host "Push to main failed, trying master..." -ForegroundColor Yellow
}

# If main failed, try master
if (-not $pushSuccess) {
    Write-Host "Attempting to push to master..." -ForegroundColor Cyan
    try {
        & git push space master --force 2>&1
        $pushSuccess = $true
        Write-Host "Successfully pushed to master" -ForegroundColor Green
    } catch {
        Write-Host "Push to master also failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Done - check huggingface.co/spaces/Sujithac07/metaai-ml-platform" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
