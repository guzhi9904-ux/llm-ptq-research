param(
    [ValidateSet("MicroMix", "FourOverSix", "All")]
    [string]$Method = "All"
)

$ErrorActionPreference = "Stop"
$Fp4Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-PinnedSource {
    param(
        [string]$Url,
        [string]$Commit,
        [string]$Path
    )

    if (Test-Path -LiteralPath (Join-Path $Path ".git")) {
        $Current = git -C $Path rev-parse HEAD
        if ($Current -ne $Commit) {
            git -C $Path fetch origin $Commit
            git -C $Path checkout --detach $Commit
        }
        return
    }
    if ((Test-Path -LiteralPath $Path) -and (Get-ChildItem -LiteralPath $Path -Force)) {
        throw "依赖目录不是空目录，也不是 Git 仓库：$Path"
    }
    git clone --filter=blob:none --no-checkout $Url $Path
    git -C $Path checkout --detach $Commit
}

if ($Method -in @("MicroMix", "All")) {
    Get-PinnedSource `
        -Url "https://github.com/NVIDIA/cutlass.git" `
        -Commit "a1aaf2300a8fc3a8106a05436e1a2abad0930443" `
        -Path (Join-Path $Fp4Root "methods/MicroMix/upstream/cutlass")
}

if ($Method -in @("FourOverSix", "All")) {
    $Base = Join-Path $Fp4Root "methods/FourOverSix/upstream/third_party"
    $Dependencies = @(
        @("https://github.com/NVIDIA/cutlass.git", "ec8daf642d69fc31352ac6fa6e14a0de9019604b", "cutlass"),
        @("https://github.com/Dao-AILab/fast-hadamard-transform.git", "f134af63deb2df17e1171a9ec1ea4a7d8604d5ca", "fast-hadamard-transform"),
        @("https://github.com/jackcook/flame-fouroversix.git", "0514985a7ebe1b566955c2061382383e5e505323", "flame"),
        @("https://github.com/jackcook/fp-quant-fouroversix.git", "65eb81f124da42c00d37bf131e6c5e1fd2da9281", "fp-quant"),
        @("https://github.com/jackcook/llm-awq-fouroversix.git", "937e821154113f69ea6c2290fd54f12f544c7140", "llm-awq"),
        @("https://github.com/IST-DASLab/qutlass.git", "03c6337c51ac57a65813658cf68ca7236526a904", "qutlass"),
        @("https://github.com/jackcook/spinquant-fouroversix.git", "d7afed8dabef3e7a1c6cb748891fd36b80f10f90", "spinquant")
    )
    foreach ($Dependency in $Dependencies) {
        Get-PinnedSource `
            -Url $Dependency[0] `
            -Commit $Dependency[1] `
            -Path (Join-Path $Base $Dependency[2])
    }
}

Write-Output "FP4 外部依赖已固定到清单中的 commit。"
