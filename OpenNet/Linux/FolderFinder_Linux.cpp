
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Linux/FolderFinder_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <stdio.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// ===== OpenNet/Linux ======================================================
#include "../FolderFinder.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class FolderFinder_Linux : public FolderFinder
{

public:

    FolderFinder_Linux();

};

// Static variable
/////////////////////////////////////////////////////////////////////////////

static FolderFinder_Linux sFolderFinder_Linux;

// Public
/////////////////////////////////////////////////////////////////////////////

FolderFinder_Linux::FolderFinder_Linux()
{
    sprintf( mInstallFolder, "/usr/local/OpenNet_%u.%u", VERSION_MAJOR, VERSION_MINOR );

    sprintf( mBinaryFolder , "%s/bin", mInstallFolder);
    sprintf( mIncludeFolder, "%s/inc", mInstallFolder);
    sprintf( mLibraryFolder, "%s/lib", mInstallFolder);

    gFolderFinder = this;
}
