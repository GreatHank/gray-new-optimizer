$ErrorActionPreference = "Stop"

$python = "C:\Users\18441\anaconda3\python.exe"
$optimizer = "coding\python\order_decoupling_grayscale.py"
$target = "output\easy_picture_3x3_order_analysis\easy_picture_3x3_target.mat"
$common = @(
    $optimizer,
    "--mat-file", $target,
    "--seed", "42",
    "--channel-count", "9",
    "--order-grid-size", "3",
    "--order-m-start", "-3",
    "--order-n-start", "-3",
    "--image-loss-mode", "energy",
    "--device", "cuda",
    "--log-interval", "100"
)

function Invoke-Stage {
    param([string[]]$Arguments)
    & $python @common @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Optimization stage failed with exit code $LASTEXITCODE"
    }
}

switch ($args[0]) {
    "A100" {
        Invoke-Stage @("--output-dir", "output\easy_picture_3x3_stage_a_main_100", "--epochs", "100", "--lr", "1e-3")
    }
    "A1000" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_a_main_total_1000",
            "--epochs", "900", "--lr", "1e-3",
            "--initial-results", "output\easy_picture_3x3_stage_a_main_100\optimized_results.npz"
        )
    }
    "B" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_b_weak_structure_500",
            "--epochs", "500", "--lr", "5e-4",
            "--initial-results", "output\easy_picture_3x3_stage_a_main_total_1000\optimized_results.npz",
            "--priority-channels", "6", "8", "9", "--priority-channel-weight", "3"
        )
    }
    "C" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_c_brightness_cv_300",
            "--epochs", "300", "--lr", "2e-4",
            "--initial-results", "output\easy_picture_3x3_stage_b_weak_structure_500\optimized_results.npz",
            "--priority-channels", "6", "8", "9", "--priority-channel-weight", "3",
            "--brightness-consistency-weight", "10", "--worst-channel-weight", "5"
        )
    }
    "D" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_d_level_cv_300",
            "--epochs", "300", "--lr", "1e-4",
            "--initial-results", "output\easy_picture_3x3_stage_c_brightness_cv_300\optimized_results.npz",
            "--priority-channels", "6", "8", "9", "--priority-channel-weight", "3",
            "--brightness-consistency-weight", "10", "--worst-channel-weight", "5",
            "--cross-level-weight", "20"
        )
    }
    "E" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_e_background_300",
            "--epochs", "300", "--lr", "1e-4",
            "--initial-results", "output\easy_picture_3x3_stage_d_level_cv_300\optimized_results.npz",
            "--priority-channels", "6", "8", "9", "--priority-channel-weight", "3",
            "--brightness-consistency-weight", "10", "--worst-channel-weight", "5",
            "--cross-level-weight", "20",
            "--background-uniformity-weight", "5",
            "--background-cluster-weight", "2", "--background-cluster-kernel", "9",
            "--background-cluster-upper", "0.9"
        )
    }
    "F" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_f_snr15_500",
            "--epochs", "500", "--lr", "1e-4",
            "--initial-results", "output\easy_picture_3x3_stage_e_background_300\optimized_results.npz",
            "--priority-channels", "6", "8", "9", "--priority-channel-weight", "3",
            "--brightness-consistency-weight", "10", "--worst-channel-weight", "5",
            "--cross-level-weight", "20",
            "--background-uniformity-weight", "5",
            "--background-cluster-weight", "2", "--background-cluster-kernel", "9",
            "--background-cluster-upper", "0.9",
            "--paper-snr-weight", "20", "--paper-snr-target-db", "15"
        )
    }
    "FLIMIT" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_f_snr_limit_1000",
            "--epochs", "1000", "--lr", "2e-4",
            "--initial-results", "output\easy_picture_3x3_stage_f_snr15_500\optimized_results.npz",
            "--paper-snr-weight", "100", "--paper-snr-target-db", "15"
        )
    }
    "FLIMIT2" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_f_snr_limit_total_2500",
            "--epochs", "1500", "--lr", "2e-4",
            "--initial-results", "output\easy_picture_3x3_stage_f_snr_limit_1000\optimized_results.npz",
            "--paper-snr-weight", "100", "--paper-snr-target-db", "15"
        )
    }
    "FLIMIT3" {
        Invoke-Stage @(
            "--output-dir", "output\easy_picture_3x3_stage_f_snr_limit_total_3500",
            "--epochs", "1000", "--lr", "2e-4",
            "--initial-results", "output\easy_picture_3x3_stage_f_snr_limit_total_2500\optimized_results.npz",
            "--paper-snr-weight", "100", "--paper-snr-target-db", "15"
        )
    }
    default {
        throw "Specify one stage: A100, A1000, B, C, D, E, F, FLIMIT, FLIMIT2, or FLIMIT3"
    }
}
