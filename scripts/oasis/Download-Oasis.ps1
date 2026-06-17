<#
.SYNOPSIS
    Windows ARM64-friendly wrapper for the OASIS-3 / OASIS-4 download scripts.

.DESCRIPTION
    The official NrgXnat/oasis-scripts are bash. On a Snapdragon / Windows-on-ARM
    edge box you may not have Git Bash installed, so this wrapper either:

      * delegates to the vendored .sh scripts when bash.exe is found, OR
      * performs a pure-PowerShell XNAT session download (-Native) using the
        same NITRC IR REST endpoints (works for scans and freesurfer types).

    Your NITRC IR username defaults to 'hakeemadeniji' (override with -Username
    or the NITRC_USERNAME environment variable). You are prompted for the
    password securely; it is never written to disk or the command history.

.PARAMETER Type
    One of: scans, freesurfer, pup, pup-match

.PARAMETER InputCsv
    CSV of experiment/freesurfer/pup IDs (see examples/ in this folder).

.PARAMETER OutDir
    Destination directory for downloaded + extracted files.

.PARAMETER ScanType
    (scans only) e.g. T1w, FLAIR, or ALL (default).

.PARAMETER MatchString
    (pup-match only) substring to match within PUP filenames.

.PARAMETER Native
    Force the pure-PowerShell download path instead of bash.

.EXAMPLE
    ./Download-Oasis.ps1 -Type freesurfer -InputCsv examples/freesurfer_ids.csv -OutDir ../../data/oasis_freesurfer

.EXAMPLE
    ./Download-Oasis.ps1 -Type scans -InputCsv examples/scan_ids.csv -OutDir ../../data/oasis_scans -ScanType T1w -Native
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('scans', 'freesurfer', 'pup', 'pup-match')]
    [string]$Type,

    [Parameter(Mandatory = $true)]
    [string]$InputCsv,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [string]$Username = $(if ($env:NITRC_USERNAME) { $env:NITRC_USERNAME } else { 'hakeemadeniji' }),

    [string]$ScanType = 'ALL',

    [string]$MatchString = '',

    [switch]$Native
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Site = 'https://www.nitrc.org/ir'

function Find-Bash {
    $cmd = Get-Command bash.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:ProgramFiles\Git\bin\bash.exe", "$env:ProgramFiles\Git\usr\bin\bash.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Get-Credentials {
    Write-Host "NITRC IR username: $Username"
    $secure = Read-Host -AsSecureString "Enter your password for accessing OASIS data on NITRC IR"
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    return $plain
}

function Invoke-NativeDownload {
    param([string]$Password)

    if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

    # 1. Open an authenticated XNAT session (JSESSIONID stored in $session).
    $pair = "$Username`:$Password"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($pair))
    $headers = @{ Authorization = "Basic $b64" }
    $session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
    Write-Host "[*] Opening NITRC IR session as $Username ..."
    Invoke-RestMethod -Uri "$Site/data/JSESSION" -Headers $headers -WebSession $session -Method Get | Out-Null

    $rows = Import-Csv -Path $InputCsv
    foreach ($row in $rows) {
        $idField = ($row.PSObject.Properties | Select-Object -First 1).Value
        $id = "$idField".Trim()
        if (-not $id) { continue }

        $subject = ($id -split '_')[0]
        $project = if ($id -like 'OAS4*') { 'OASIS4' } else { 'OASIS3' }

        if ($Type -eq 'scans') {
            $experiment = $id
            $url = "$Site/data/archive/projects/$project/subjects/$subject/experiments/$experiment/scans/$ScanType/files?format=tar.gz"
        }
        elseif ($Type -eq 'freesurfer') {
            $parts = $id -split '_'
            $experiment = "$subject`_MR_$($parts[2])"
            $url = "$Site/data/archive/projects/$project/subjects/$subject/experiments/$experiment/assessors/$id/files?format=tar.gz"
        }
        else {
            Write-Warning "Native mode supports 'scans' and 'freesurfer' only. Use bash for '$Type'."
            return
        }

        $tar = Join-Path $OutDir "$id.tar.gz"
        Write-Host "[*] Downloading $id -> $tar"
        try {
            Invoke-WebRequest -Uri $url -WebSession $session -OutFile $tar -Headers @{ Expect = '' }
        } catch {
            Write-Warning "Failed to download $id : $_"
            continue
        }

        # 2. Validate + extract with native Windows tar.
        & tar -tf $tar > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[+] Extracting $id ..."
            & tar -xzf $tar -C $OutDir
            Remove-Item $tar -Force
        } else {
            Write-Warning "Downloaded file for $id is not a valid archive (likely no data / bad credentials)."
        }
    }

    # 3. Close the session.
    Invoke-RestMethod -Uri "$Site/data/JSESSION" -WebSession $session -Method Delete | Out-Null
    Write-Host "[SUCCESS] Native download complete -> $OutDir"
}

function Invoke-ViaBash {
    param([string]$BashPath)
    $map = @{
        'scans'      = 'download_oasis_scans_tar.sh'
        'freesurfer' = 'download_oasis_freesurfer_tar.sh'
        'pup'        = 'download_oasis_pup_tar.sh'
        'pup-match'  = 'download_oasis_pup_files_by_partial_filename_match.sh'
    }
    $script = Join-Path $ScriptDir $map[$Type]
    Write-Host "[*] Delegating to bash: $script"
    if ($Type -eq 'scans') {
        & $BashPath $script $InputCsv $OutDir $Username $ScanType
    } elseif ($Type -eq 'pup-match') {
        & $BashPath $script $InputCsv $OutDir $Username $MatchString
    } else {
        & $BashPath $script $InputCsv $OutDir $Username
    }
}

# --- main ---------------------------------------------------------------
$bash = Find-Bash
if ($Native -or -not $bash) {
    if (-not $bash) { Write-Host "[i] bash.exe not found; using native PowerShell download path." }
    $pw = Get-Credentials
    Invoke-NativeDownload -Password $pw
} else {
    Invoke-ViaBash -BashPath $bash
}
