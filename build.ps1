param(
    [string]$Action = "build"
)

# Simple PowerShell build wrapper for CMake-based build
# Usage: .\build.ps1 -Action build|clean|rebuild|run

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$buildDir = Join-Path $root "build"

function Configure-Build {
    Write-Host "Configuring CMake..."
    cmake -S $root -B $buildDir
}

function Build {
    Write-Host "Building project..."
    cmake --build $buildDir --config Release
}

function Clean {
    if (Test-Path $buildDir) {
        Write-Host "Removing build directory..."
        Remove-Item $buildDir -Recurse -Force
    } else {
        Write-Host "No build directory to remove."
    }
}

function Run {
    $exe = Join-Path $buildDir "Gauss_1D.exe"
    if (Test-Path $exe) {
        Write-Host "Running example..."
        & $exe
    } else {
        Write-Host "Executable not found. Build first."
    }
}

switch ($Action) {
    'build' { Configure-Build; Build }
    'clean' { Clean }
    'rebuild' { Clean; Configure-Build; Build }
    'run' { Run }
    default { Write-Host "Unknown action: $Action" }
}
