$ErrorActionPreference = "Continue"
$log   = "C:\Users\lenovo\Desktop\nndl_train.log"
$root  = "C:\Users\lenovo\Desktop\NNDLPJ"
$codes = Join-Path $root "codes"

function Append-Log($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Out-File -Append -FilePath $log -Encoding utf8
}

Append-Log "=== RELAUNCHED (detached background runner) ==="

# Phase 1: full study (resumes automatically)
Set-Location $codes
Append-Log "phase1: python run_full_study.py --pack all --epochs-mlp 30 --epochs-cnn 20 --epochs-direction 8"
& python -u run_full_study.py --pack all --epochs-mlp 30 --epochs-cnn 20 --epochs-direction 8 --valid-size 10000 *>&1 | Out-File -Append -FilePath $log -Encoding utf8
$rc1 = $LASTEXITCODE
Append-Log "phase1: pack-all exit=$rc1"

# Phase 2: build LaTeX report
$rc2 = -1
if ($rc1 -eq 0) {
    Set-Location $root
    Append-Log "phase2: python codes/build_report.py"
    & python -u codes/build_report.py *>&1 | Out-File -Append -FilePath $log -Encoding utf8
    $rc2 = $LASTEXITCODE
    Append-Log "phase2: build_report exit=$rc2"
}

# Phase 3: compile PDF (two passes for cross references)
if ($rc2 -eq 0) {
    Set-Location $root
    Append-Log "phase3a: pdflatex pass 1"
    & pdflatex -interaction=nonstopmode -halt-on-error MNIST_From_Scratch_Report_Leyan_Huang.tex *>&1 | Out-File -Append -FilePath $log -Encoding utf8
    Append-Log "phase3a: pdflatex pass 1 exit=$LASTEXITCODE"
    Append-Log "phase3b: pdflatex pass 2"
    & pdflatex -interaction=nonstopmode -halt-on-error MNIST_From_Scratch_Report_Leyan_Huang.tex *>&1 | Out-File -Append -FilePath $log -Encoding utf8
    Append-Log "phase3b: pdflatex pass 2 exit=$LASTEXITCODE"
    if (Test-Path "MNIST_From_Scratch_Report_Leyan_Huang.pdf") {
        $pdf = Get-Item "MNIST_From_Scratch_Report_Leyan_Huang.pdf"
        Append-Log "phase3: PDF produced, size=$([math]::Round($pdf.Length/1KB,1))KB"
    }
}

Append-Log "=== ALL DONE ==="
