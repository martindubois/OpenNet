
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/FolderFinder.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <memory.h>

// ===== OpenNet ============================================================
#include "FolderFinder.h"

// Global variables
/////////////////////////////////////////////////////////////////////////////

const FolderFinder * gFolderFinder = NULL;

// Public
/////////////////////////////////////////////////////////////////////////////

const char * FolderFinder::GetBinaryFolder() const
{
    return mBinaryFolder;
}

const char * FolderFinder::GetDriverFolder() const
{
    return mDriverFolder;
}

const char * FolderFinder::GetIncludeFolder() const
{
    return mIncludeFolder;
}

const char * FolderFinder::GetInstallFolder() const
{
    return mInstallFolder;
}

const char * FolderFinder::GetLibraryFolder() const
{
    return mLibraryFolder;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

FolderFinder::FolderFinder()
{
    memset(&mBinaryFolder , 0, sizeof(mBinaryFolder ));
    memset(&mBinaryFolder , 0, sizeof(mDriverFolder ));
    memset(&mIncludeFolder, 0, sizeof(mIncludeFolder));
    memset(&mInstallFolder, 0, sizeof(mInstallFolder));
    memset(&mLibraryFolder, 0, sizeof(mLibraryFolder));
}
