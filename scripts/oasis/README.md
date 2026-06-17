# OASIS-3 / OASIS-4 Download Scripts

These scripts pull real **OASIS-3** and **OASIS-4** imaging derivatives from
NITRC IR (`https://www.nitrc.org/ir`) so the pipeline can run on production-grade,
regionally-segmented data instead of the bundled 2D OASIS-1 slices.

They are vendored **verbatim** from the official
[NrgXnat/oasis-scripts](https://github.com/NrgXnat/oasis-scripts) repository so
behaviour matches the upstream tooling exactly. A Windows ARM64 PowerShell
wrapper (`Download-Oasis.ps1`) is added on top for Snapdragon edge boxes.

## Prerequisites

1. **NITRC IR account with OASIS access.** Your username is **`hakeemadeniji`**.
   You will be prompted for your password at runtime (never stored).
2. Apply for OASIS-3 / OASIS-4 access at <https://sites.wustl.edu/oasisbrains/>
   and accept the Data Use Agreement; access is granted on your NITRC account.
3. `curl` and `tar` — both ship with Windows 11 and Git Bash.

## What each script downloads

| Script | Downloads | ID column example |
|--------|-----------|-------------------|
| `download_oasis_scans_tar.sh` | Raw MRI/PET scans (NIfTI) | `OAS30001_MR_d0129` |
| `download_oasis_freesurfer_tar.sh` | FreeSurfer derivatives (incl. `stats/aseg.stats`) | `OAS30001_Freesurfer53_d0129` |
| `download_oasis_pup_tar.sh` | PET Unified Pipeline (PUP) outputs | `OAS30001_AV45_PUPTIMECOURSE_d2430` |
| `download_oasis_pup_files_by_partial_filename_match.sh` | Individual PUP files matching a string | `OAS30001_AV45_PUPTIMECOURSE_d2430` |

> The **Regional Volumetry Agent** consumes the FreeSurfer `stats/aseg.stats`
> files produced by `download_oasis_freesurfer_tar.sh`. Point `FREESURFER_ROOT`
> in your `.env` at the output directory (default `data/oasis_freesurfer`).

## Usage — Git Bash (Linux/Mac/WSL/Git Bash)

```bash
cd scripts/oasis
chmod +x *.sh

# FreeSurfer derivatives (feeds regional volumetry)
./download_oasis_freesurfer_tar.sh examples/freesurfer_ids.csv ../../data/oasis_freesurfer hakeemadeniji

# T1w structural scans only
./download_oasis_scans_tar.sh examples/scan_ids.csv ../../data/oasis_scans hakeemadeniji T1w

# PUP PET outputs
./download_oasis_pup_tar.sh examples/pup_ids.csv ../../data/oasis_pup hakeemadeniji

# Specific PUP files (e.g. SUVR images)
./download_oasis_pup_files_by_partial_filename_match.sh examples/pup_ids.csv ../../data/oasis_pup hakeemadeniji "suvr"
```

## Usage — Windows ARM64 (PowerShell, no Git Bash required)

`Download-Oasis.ps1` runs the `.sh` scripts when Git Bash is present, otherwise
falls back to a pure-PowerShell XNAT session download (scans + freesurfer):

```powershell
cd scripts\oasis

# FreeSurfer (native PowerShell path, Snapdragon-friendly)
./Download-Oasis.ps1 -Type freesurfer -InputCsv examples\freesurfer_ids.csv -OutDir ..\..\data\oasis_freesurfer -Native

# T1w scans
./Download-Oasis.ps1 -Type scans -InputCsv examples\scan_ids.csv -OutDir ..\..\data\oasis_scans -ScanType T1w
```

Username defaults to `hakeemadeniji` (override with `-Username` or
`$env:NITRC_USERNAME`).

## Building the input CSVs

Export the session/freesurfer/pup IDs you want from the OASIS data dashboards on
NITRC IR, or copy from the OASIS data-dictionary spreadsheets. The CSV must be
Unix line-endings with a single header row (see the templates in `examples/`).

## Security note

Credentials are read interactively and used only to open a short-lived XNAT
`JSESSION`, which is deleted when the script finishes. Do not commit downloaded
PHI-adjacent imaging into the repo — `data/oasis_freesurfer/`, `data/oasis_scans/`
and `data/oasis_pup/` are git-ignored.
