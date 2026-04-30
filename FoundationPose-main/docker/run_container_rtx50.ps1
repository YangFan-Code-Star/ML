param(
  [string]$Image = "foundationpose-rtx50:latest",
  [string]$Name = "foundationpose-rtx50",
  [string]$Command = "bash",
  [switch]$Build,
  [switch]$Keep
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

if ($Build) {
  docker build -t $Image -f (Join-Path $scriptDir "Dockerfile.rtx50") $repoRoot
}

$existing = docker ps -a --filter "name=^/$Name$" --format "{{.ID}}"
if ($existing) {
  docker rm -f $Name | Out-Null
}

$removeFlag = "--rm"
if ($Keep) {
  $removeFlag = ""
}

docker run `
  $removeFlag `
  --gpus all `
  --env NVIDIA_DISABLE_REQUIRE=1 `
  --env TORCH_CUDA_ARCH_LIST="12.0" `
  --ipc=host `
  --name $Name `
  -v "${repoRoot}:/workspace/FoundationPose" `
  -w /workspace/FoundationPose `
  $Image `
  bash -lc $Command
