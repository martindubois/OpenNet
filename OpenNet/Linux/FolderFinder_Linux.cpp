
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Linux/FolderFinder_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <link.h>
#include <stdio.h>

// ===== OpenNet/Linux ======================================================
#include "../FolderFinder.h"

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static int ProcessSO( struct dl_phdr_info * aInfo, size_t aSize_byte, void * aData );

// Class
/////////////////////////////////////////////////////////////////////////////

class FolderFinder_Linux : public FolderFinder
{

public:

    FolderFinder_Linux();

    void SetBinaryFolder( const char * aPath, unsigned int aLength );

};

// Static variable
/////////////////////////////////////////////////////////////////////////////

static FolderFinder_Linux sFolderFinder_Linux;

// Public
/////////////////////////////////////////////////////////////////////////////

FolderFinder_Linux::FolderFinder_Linux()
{
    memset( & mBinaryFolder , 0, sizeof( mBinaryFolder  ) );
    memset( & mIncludeFolder, 0, sizeof( mIncludeFolder ) );
    memset( & mInstallFolder, 0, sizeof( mInstallFolder ) );
    memset( & mLibraryFolder, 0, sizeof( mLibraryFolder ) );

    int lRet = dl_iterate_phdr( ProcessSO, this );
    assert( 0 == lRet );
}

void FolderFinder_Linux::SetBinaryFolder( const char * aPath, unsigned int aLength )
{
    assert( NULL != aPath );

    assert( NULL == gFolderFinder );

    memcpy( mBinaryFolder, aPath, aLength );

    unsigned int lLength = aLength - 4;
    if ( 0 == strcmp( "/bin", aPath + lLength ) )
    {
        memcpy( mInstallFolder, aPath, lLength );
    }
    else
    {
        memcpy( mInstallFolder , aPath, aLength );
    }

    sprintf( mIncludeFolder, "%s/inc", mInstallFolder );
    sprintf( mLibraryFolder, "%s/lib", mInstallFolder );

    gFolderFinder = this;
}

// Static function
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static int ProcessSO( struct dl_phdr_info * aInfo, size_t aSize_byte, void * aData )
{
    assert( NULL                          != aInfo      );
    assert( sizeof( struct dl_phdr_info ) <= aSize_byte );
    assert( NULL                          != aData      );

    unsigned int lLength = strlen( aInfo->dlpi_name );
    if ( 14 <= lLength )
    {
        lLength -= 14;

        if ( 0 == strcmp( "/libOpenNet.so", aInfo->dlpi_name + lLength ) )
        {
            FolderFinder_Linux * lThis = reinterpret_cast< FolderFinder_Linux * >( aData );

            lThis->SetBinaryFolder( aInfo->dlpi_name, lLength );
        }
    }

    return 0;
}
