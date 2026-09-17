<#
.SYNOPSIS
Re-export the authoritative, author-edited panel PPTX without regenerating it.
.DESCRIPTION
Requires Windows PowerPoint and Python with PyMuPDF >= 1.24. The PPT is opened
read-only and never saved. Only the opened presentation is closed; PowerPoint
and other user presentations are never quit or closed. Exact embedded images
are restored into the native PDF before canonical hashes/previews are updated.
#>
[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$Python = 'python'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ScopedPath {
    param([string]$Folder, [string]$RelativePath)
    $taskFolder = [System.IO.Path]::GetFullPath($Folder).TrimEnd('\', '/')
    $taskCandidate = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($taskFolder, $RelativePath)
    )
    if (-not $taskCandidate.StartsWith(
        $taskFolder + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Path must stay inside the panel source package: $RelativePath"
    }
    return $taskCandidate
}

$taskSourceDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
$taskBundleRoot = [System.IO.Path]::GetFullPath((Join-Path $taskSourceDir '..'))
$taskManifestPath = Join-Path $taskSourceDir 'export_manifest.json'
$taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw | ConvertFrom-Json
$taskPptx = Get-ScopedPath $taskSourceDir ([string]$taskManifest.source_pptx)
$taskCanonicalPdf = Get-ScopedPath $taskBundleRoot ([string]$taskManifest.exported_pdf)
$taskRestoreScript = Join-Path $taskSourceDir 'restore_source_images.py'
$taskPanelStem = [System.IO.Path]::GetFileNameWithoutExtension($taskPptx).ToLowerInvariant()
$taskBuildScript = Join-Path $taskBundleRoot ('code/build_' + $taskPanelStem + '.py')

foreach ($taskRequired in @($taskPptx, $taskCanonicalPdf, $taskRestoreScript, $taskBuildScript)) {
    if (-not (Test-Path -LiteralPath $taskRequired -PathType Leaf)) {
        throw "Missing required panel file: $taskRequired"
    }
}
$null = Get-Command $Python -ErrorAction Stop
& $Python -X utf8 -c "import fitz; assert tuple(map(int, fitz.VersionBind.split('.')[:2])) >= (1, 24), 'PyMuPDF >= 1.24 is required'"
if ($LASTEXITCODE -ne 0) {
    throw 'Python preflight failed; install the figure requirements or pass -Python with the correct interpreter.'
}

$taskSourceHashBefore = (Get-FileHash -LiteralPath $taskPptx -Algorithm SHA256).Hash.ToLowerInvariant()
$taskTempDir = Join-Path ([System.IO.Path]::GetTempPath()) (
    'figure-panel-export-' + [System.Guid]::NewGuid().ToString('N')
)
$null = New-Item -Path $taskTempDir -ItemType Directory
$taskNativePdf = Join-Path $taskTempDir 'native.pdf'
$taskRestoredPdf = Join-Path $taskTempDir 'restored.pdf'
$taskRecordPath = Join-Path $taskTempDir 'restoration.json'
$taskNextManifest = Join-Path $taskTempDir 'export_manifest.json'
$taskPreviousPdf = Join-Path $taskTempDir 'previous.pdf'
$taskPreviousManifest = Join-Path $taskTempDir 'previous_manifest.json'
$taskSucceeded = $false

try {
    $taskPowerPoint = $null
    $taskPresentations = $null
    $taskOpenedPresentation = $null
    try {
        $taskPowerPoint = New-Object -ComObject PowerPoint.Application
        $taskPresentations = $taskPowerPoint.Presentations
        # Refuse to reuse/close a source deck the user already has open.
        for ($taskIndex = 1; $taskIndex -le $taskPresentations.Count; $taskIndex++) {
            $taskExistingPresentation = $taskPresentations.Item($taskIndex)
            try {
                if ([string]::Equals(
                    [string]$taskExistingPresentation.FullName, $taskPptx,
                    [System.StringComparison]::OrdinalIgnoreCase
                )) {
                    throw 'This source PPT is already open. Save and close that presentation yourself, then rerun; no user presentation was closed.'
                }
            }
            finally {
                [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskExistingPresentation)
            }
        }
        # ReadOnly = msoTrue; Untitled = msoFalse; WithWindow = msoFalse.
        $taskOpenedPresentation = $taskPresentations.Open($taskPptx, -1, 0, 0)
        # ppSaveAsPDF = 32. No PPT Save/SaveAs and no design-generation code.
        $taskOpenedPresentation.SaveAs($taskNativePdf, 32)
    }
    finally {
        if ($null -ne $taskOpenedPresentation) {
            try { $taskOpenedPresentation.Close() }
            finally {
                [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskOpenedPresentation)
            }
        }
        if ($null -ne $taskPresentations) {
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskPresentations)
        }
        if ($null -ne $taskPowerPoint) {
            # Never call Application.Quit: the instance may contain user decks.
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($taskPowerPoint)
        }
    }
    if (-not (Test-Path -LiteralPath $taskNativePdf -PathType Leaf)) {
        throw 'PowerPoint did not produce the temporary native PDF.'
    }

    & $Python -X utf8 $taskRestoreScript $taskPptx $taskNativePdf $taskRestoredPdf --record $taskRecordPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Image restoration failed. The canonical PDF and manifest have not been replaced.'
    }
    $taskSourceHashAfter = (Get-FileHash -LiteralPath $taskPptx -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskSourceHashAfter -ne $taskSourceHashBefore) {
        throw 'The PPT source changed during export. Canonical files were not replaced; repeat after editing is complete.'
    }
    $taskRecord = Get-Content -LiteralPath $taskRecordPath -Raw | ConvertFrom-Json
    $taskActualPdfHash = (Get-FileHash -LiteralPath $taskRestoredPdf -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskRecord.source_pptx_sha256 -ne $taskSourceHashAfter -or
        $taskRecord.exported_pdf_sha256 -ne $taskActualPdfHash) {
        throw 'Restoration hashes do not match the source/export.'
    }

    # Keep the existing source_pptx, exported_pdf and provenance fields.
    $taskManifest.source_pptx_sha256 = $taskRecord.source_pptx_sha256
    $taskManifest.exported_pdf_sha256 = $taskRecord.exported_pdf_sha256
    $taskManifest.image_restoration = @($taskRecord.image_restoration)
    $taskManifest | Add-Member -NotePropertyName exported_at_utc -NotePropertyValue (
        [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    ) -Force
    [System.IO.File]::WriteAllText(
        $taskNextManifest,
        (($taskManifest | ConvertTo-Json -Depth 30) + [Environment]::NewLine),
        [System.Text.UTF8Encoding]::new($false)
    )

    # Bound replacements to the validated canonical pair; retain temporary
    # previous copies so a failed second replacement can restore the pair.
    Copy-Item -LiteralPath $taskCanonicalPdf -Destination $taskPreviousPdf
    Copy-Item -LiteralPath $taskManifestPath -Destination $taskPreviousManifest
    try {
        Copy-Item -LiteralPath $taskRestoredPdf -Destination $taskCanonicalPdf -Force
        Copy-Item -LiteralPath $taskNextManifest -Destination $taskManifestPath -Force
    }
    catch {
        Copy-Item -LiteralPath $taskPreviousPdf -Destination $taskCanonicalPdf -Force
        Copy-Item -LiteralPath $taskPreviousManifest -Destination $taskManifestPath -Force
        throw
    }
    & $Python -X utf8 $taskBuildScript
    if ($LASTEXITCODE -ne 0) {
        throw 'The PDF and manifest were updated, but preview regeneration failed; rerun the panel build script.'
    }
    $taskSucceeded = $true
    Write-Output "Updated native PDF, restored full-resolution images, manifest and previews: $taskCanonicalPdf"
    Write-Output "Authoritative PPT source unchanged: $taskPptx"
}
finally {
    if ($taskSucceeded) {
        # Explicit files only; no recursive or broad-directory deletion.
        foreach ($taskTempFile in @(
            $taskNativePdf, $taskRestoredPdf, $taskRecordPath, $taskNextManifest,
            $taskPreviousPdf, $taskPreviousManifest
        )) {
            if (Test-Path -LiteralPath $taskTempFile -PathType Leaf) {
                Remove-Item -LiteralPath $taskTempFile -Force
            }
        }
        Remove-Item -LiteralPath $taskTempDir
    }
    else {
        Write-Warning "Temporary export files retained for diagnosis: $taskTempDir"
    }
}
