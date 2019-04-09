
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Windows/FolderFinder_Windows.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdio.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet/Windows ====================================================
#include "../FolderFinder.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class FolderFinder_Windows : public FolderFinder
{

public:

    FolderFinder_Windows();

    virtual const char * GetDriverFolder() const;

private:

    char mDriverFolder[ 1024 ];
    
};

// Static variable
/////////////////////////////////////////////////////////////////////////////

static FolderFinder_Windows sFolderFinder_Windows;

// Public
/////////////////////////////////////////////////////////////////////////////

FolderFinder_Windows::FolderFinder_Windows()
{
    memset(&mBinaryFolder , 0, sizeof(mDriverFolder ));

    DWORD lRet = GetModuleFileNameA(NULL, mBinaryFolder, sizeof(mBinaryFolder));
    assert(0 < lRet);

    char * lPtr = strrchr(mBinaryFolder, '\\');
    assert(NULL != lPtr);

    (*lPtr) = '\0';

    strncpy_s(mInstallFolder, mBinaryFolder, sizeof(mInstallFolder) - 1);

    sprintf_s(mDriverFolder , "%s\\Drivers"  , mInstallFolder);
    sprintf_s(mIncludeFolder, "%s\\Includes" , mInstallFolder);
    sprintf_s(mLibraryFolder, "%s\\Libraries", mInstallFolder);

    gFolderFinder = this;
}

// ===== FolderFinder =======================================================

const char * FolderFinder_Windows::GetDriverFolder() const
{
    return mDriverFolder;
}
