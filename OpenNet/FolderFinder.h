
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/FolderFinder.h

#pragma once

// Class
/////////////////////////////////////////////////////////////////////////////

class FolderFinder
{

public:

    virtual const char * GetBinaryFolder () const;
    virtual const char * GetIncludeFolder() const;
    virtual const char * GetInstallFolder() const;
    virtual const char * GetLibraryFolder() const;

    #ifdef _KMS_WINDOWS_
        virtual const char * GetDriverFolder () const = 0;
    #endif

protected:

    FolderFinder();

    char mBinaryFolder [1024];
    char mIncludeFolder[1024];
    char mInstallFolder[1020];
    char mLibraryFolder[1024];

};

// Global variables
/////////////////////////////////////////////////////////////////////////////

extern const FolderFinder * gFolderFinder;
