
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
    assert( 1 == lRet );
}

void FolderFinder_Linux::SetBinaryFolder( const char * aPath, unsigned int aLength )
{
    assert( NULL != aPath );

    assert( NULL == gFolderFinder );

    memcpy( mBinaryFolder, aPath, aLength );

    if ( 0 == strncmp( "./Binaries", aPath, aLength ) )
    {
        // We are running automated tests in an OpenNet project folder
        strcpy( mIncludeFolder, "./Includes"  );
        strcpy( mInstallFolder, "."           );
        strcpy( mLibraryFolder, "./Libraries" );
    }
    else
    {
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

    }

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
    if ( 13 <= lLength )
    {
        lLength -= 13;

        if ( 0 == strcmp( "libOpenNet.so", aInfo->dlpi_name + lLength ) )
        {
            FolderFinder_Linux * lThis = reinterpret_cast< FolderFinder_Linux * >( aData );

            if ( 0 < lLength )
            {
                assert( '/' == aInfo->dlpi_name[ lLength - 1 ] );

                lThis->SetBinaryFolder( aInfo->dlpi_name, lLength - 1 );
            }
            else
            {
                // We run into the folder where libOpenNet.so is.
                char lCurrent[ 4096 ];

                char * lRet = getcwd( lCurrent, sizeof( lCurrent ) );
                assert( lCurrent == lRet );

                lLength = strlen( lCurrent );
                assert( 0 < lLength );

                lThis->SetBinaryFolder( lCurrent, lLength );
            }

            return 1;
        }
    }

    return 0;
}
