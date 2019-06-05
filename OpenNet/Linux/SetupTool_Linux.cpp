
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Linux/SetupTool_Linux.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== C ==================================================================
#include <memory.h>

// ===== OpenNet/Linux ======================================================
#include "../SetupTool_Text.h"

#include "SetupTool_Linux.h"

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    const char   * mName    ;
    unsigned short mVendorId;
    unsigned short mDeviceId;

    SetupTool_Linux::Module_Type mType;

    bool mFirst;
}
Module_Info;

// Constants
/////////////////////////////////////////////////////////////////////////////

#define ETC_MOD_PROBE_D "/etc/modprobe.d"

#define ONK_PRO1000 "ONK_Pro1000"

#define TMP_LS_MOD             "/tmp/OpenNet_lsmod.txt"
#define TMP_LS_PCI             "/tmp/OpenNet_lspci.txt"
#define TMP_MOD_PROBE          "/tmp/OpenNet_modprobe.txt"
#define TMP_UPDATE_INIT_RAM_FS "/tmp/OpenNet_update-initramfs.txt"

static const Module_Info MODULES[] =
{
    { "igb"      , 0x8086, 0x10c9, SetupTool_Linux::MOD_TYPE_NET     , true  },
    { "ixgbe"    , 0x8086, 0x10fb, SetupTool_Linux::MOD_TYPE_NET     , true  },
    { ONK_PRO1000, 0x8086, 0x10c9, SetupTool_Linux::MOD_TYPE_OPEN_NET, true  },
    { ONK_PRO1000, 0x8086, 0x10fb, SetupTool_Linux::MOD_TYPE_OPEN_NET, false },
};

#define MODULE_COUNT ( sizeof( MODULES ) / sizeof( MODULES[ 0 ] ) )

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Mod ================================================================
static const Module_Info * Mod_Find ( const char * aName, unsigned short aVendorId, unsigned short aDeviceId );
static const Module_Info & Mod_Find (                     unsigned short aVendorId, unsigned short aDeviceId );
static bool                Mod_Known( const char * aName );

// Public
/////////////////////////////////////////////////////////////////////////////

SetupTool_Linux::SetupTool_Linux( bool aDebug ) : SetupTool_Internal( aDebug )
{
    Infos_Retrieve();
}

// ===== OprnNet::SetupTool =============================================

OpenNet::Status SetupTool_Linux::Install()
{
    return OpenNet::STATUS_OK;
}

OpenNet::Status SetupTool_Linux::Uninstall()
{
    OpenNet::Status lResult = OpenNet::STATUS_OK;

    for ( unsigned int i = 0; i < MODULE_COUNT; i ++ )
    {
        const Module_Info & lMI = MODULES[ i ];

        if ( lMI.mFirst )
        {
            switch ( lMI.mType )
            {
            case MOD_TYPE_NET :
                Mod_BlackList_Disable( lMI.mName );
                lResult = OpenNet::STATUS_REBOOT_REQUIRED;
                break;

            case MOD_TYPE_OPEN_NET :
                Mod_Probe_Disable( lMI.mName );
                lResult = OpenNet::STATUS_REBOOT_REQUIRED;
                break;

            default : assert( false );
            }
        }
    }

    return lResult;
}

OpenNet::Status SetupTool_Linux::Interactif_ExecuteCommand( unsigned int aCommand )
{
    if ( mInfos.size() <= aCommand )
    {
        return OpenNet::STATUS_INVALID_COMMAND_INDEX;
    }

    OpenNet::Status lResult = OpenNet::STATUS_OK;

    const Info & lInfo = Info_Get( aCommand );

    const Module_Info & lMI = Mod_Find( lInfo.mVendorId, lInfo.mDeviceId );
    assert( NULL != ( & lMI ) );

    switch ( lInfo.mType )
    {
    case MOD_TYPE_NONE :
        Mod_Probe_Enable( ONK_PRO1000 );
        Infos_Retrieve();
        break;

    case MOD_TYPE_NET :
        Mod_BlackList_Enable( lInfo.mModule );
        Mod_Probe_Enable    ( ONK_PRO1000   );
        lResult = OpenNet::STATUS_REBOOT_REQUIRED;
        break;

    case MOD_TYPE_OPEN_NET :
        assert( NULL != lMI.mName );

        Mod_BlackList_Disable( lMI.mName );
        lResult = OpenNet::STATUS_REBOOT_REQUIRED;
        break;

    default : assert( false );
    }
}

unsigned int SetupTool_Linux::Interactif_GetCommandCount()
{
    return mInfos.size();
}

const char * SetupTool_Linux::Interactif_GetCommandText(unsigned int aCommand)
{
    if ( mInfos.size() <= aCommand )
    {
        return NULL;
    }

    const Info & lInfo = Info_Get( aCommand );
    assert( NULL != ( & lInfo )  );
    assert(    0 <  lInfo.mCount );
    assert( NULL != lInfo.mName  );

    switch ( lInfo.mType )
    {
    case MOD_TYPE_NET  :
    case MOD_TYPE_NONE :
        if ( 1 == lInfo.mCount )
        {
            sprintf( mText, "Install the OpenNet driver for the\n    %s", lInfo.mName );
        }
        else
        {
            sprintf( mText, "Install the OpenNet driver for the %u\n    %s", lInfo.mCount, lInfo.mName );
        }
        break;

    case MOD_TYPE_OPEN_NET :
        if ( 1 == lInfo.mCount )
        {
            sprintf( mText, "Uninstall the OpenNet driver for the\n    %s", lInfo.mName );
        }
        else
        {
            sprintf( mText, "Uninstall the OpenNet driver for the %u\n    %s", lInfo.mCount, lInfo.mName );
        }
        break;

    default : return NULL;
    }

    return mText;
}

OpenNet::Status SetupTool_Linux::Wizard_ExecutePage(unsigned int * aPage, unsigned int aButton)
{
    if ( mInfos.size() <= ( * aPage ) )
    {
        return OpenNet::STATUS_INVALID_PAGE_INDEX;
    }

    OpenNet::Status lResult = OpenNet::STATUS_OK;

    const Info & lInfo = Info_Get( * aPage );

    const Module_Info & lMI = Mod_Find( lInfo.mVendorId, lInfo.mDeviceId );
    assert( NULL != ( & lMI ) );

    switch ( lInfo.mType )
    {
    case MOD_TYPE_NONE :
        switch ( aButton )
        {
        case 0 :
            Mod_Probe_Enable( ONK_PRO1000 );
            Infos_Retrieve();
            break;
        case 1 : break;

        default : lResult = OpenNet::STATUS_INVALID_BUTTON_INDEX;
        }
        ( * aPage ) ++;
        break;

    case MOD_TYPE_NET :
        switch ( aButton )
        {
        case 0 :
            Mod_BlackList_Enable( lInfo.mModule );
            Mod_Probe_Enable    ( ONK_PRO1000   );
            lResult = OpenNet::STATUS_REBOOT_REQUIRED;
            break;
        case 1 : ( * aPage ) ++; break;

        default : lResult = OpenNet::STATUS_INVALID_BUTTON_INDEX;
        }
        break;

    case MOD_TYPE_OPEN_NET :
        assert( NULL != lMI.mName );

        switch ( aButton )
        {
        case 0 :
            Mod_BlackList_Disable( lMI.mName );
            lResult = OpenNet::STATUS_REBOOT_REQUIRED;
            break;
        case 1 : ( * aPage ) ++; break;

        default : lResult = OpenNet::STATUS_INVALID_BUTTON_INDEX;
        }
        break;

    default : lResult = OpenNet::STATUS_INVALID_PAGE_INDEX;
    }

    return lResult;
}

unsigned int  SetupTool_Linux::Wizard_GetPageButtonCount(unsigned int aPage)
{
    if ( mInfos.size() <= aPage )
    {
        return 0;
    }

    return 2;
}

const char * SetupTool_Linux::Wizard_GetPageButtonText(unsigned int aPage, unsigned int aButton)
{
    if ( mInfos.size() > aPage )
    {
        const Info & lInfo = Info_Get( aPage );
        switch ( lInfo.mType )
        {
        case MOD_TYPE_NET  :
        case MOD_TYPE_NONE :
            switch ( aButton )
            {
            case 0 : return "Install";
            case 1 : return "Skip"   ;
            }
            break;

        case MOD_TYPE_OPEN_NET :
            switch ( aButton )
            {
            case 0 : return "Uninstall";
            case 1 : return "Skip"     ;
            }
            break;
        }
    }

    return NULL;
}

unsigned int  SetupTool_Linux::Wizard_GetPageCount()
{
    return mInfos.size();
}

const char * SetupTool_Linux::Wizard_GetPageText(unsigned int aPage)
{
    if ( mInfos.size() > aPage )
    {
        const Info & lInfo = Info_Get( aPage );
        assert( NULL != ( & lInfo )  );
        assert(    1 <= lInfo.mCount );

        switch ( lInfo.mType )
        {
        case MOD_TYPE_NET  :
            if ( 1 == lInfo.mCount )
            {
                sprintf( mText,
                    STT_NETWORK_ADAPTER_1
                    STT_NET_CONFIG
                    STT_INSTALL
                    STT_QUESTION_INSTALL,
                    lInfo.mName );
            }
            else
            {
                sprintf( mText,
                    STT_NETWORK_ADAPTERS_2
                    STT_NET_CONFIGS
                    STT_INSTALLS
                    STT_QUESTION_INSTALL,
                    lInfo.mCount, lInfo.mName );
            }
            break;

        case MOD_TYPE_NONE :
            if ( 1 == lInfo.mCount )
            {
                sprintf( mText,
                    STT_NETWORK_ADAPTER_1
                    STT_INSTALL
                    STT_QUESTION_INSTALL,
                    lInfo.mName );
            }
            else
            {
                sprintf( mText,
                    STT_NETWORK_ADAPTERS_2
                    STT_INSTALLS
                    STT_QUESTION_INSTALL,
                    lInfo.mCount, lInfo.mName );
            }
            break;

        case MOD_TYPE_OPEN_NET :
            if ( 1 == lInfo.mCount )
            {
                sprintf( mText,
                    STT_NETWORK_ADAPTER_1
                    STT_OPEN_NET_CONFIG
                    STT_UNINSTALL
                    STT_QUESTION_UNINSTALL,
                    lInfo.mName );
            }
            else
            {
                sprintf( mText,
                    STT_NETWORK_ADAPTERS_2
                    STT_OPEN_NET_CONFIGS
                    STT_UNINSTALLS
                    STT_QUESTION_UNINSTALL,
                    lInfo.mCount, lInfo.mName );
            }
            break;

        default : assert( false );
        }
        
        return mText;
    }

    return NULL;
}

const char * SetupTool_Linux::Wizard_GetPageTitle(unsigned int aPage)
{
    if ( mInfos.size() > aPage )
    {
        const Info & lInfo = Info_Get( aPage );
        assert( NULL != ( & lInfo )  );

        switch ( lInfo.mType )
        {
        case MOD_TYPE_NET  :
        case MOD_TYPE_NONE :
            return "Install OpenNet driver";

        case MOD_TYPE_OPEN_NET : return "Uninstall OpenNet driver";

        default : assert( false );
        }
    }

    return NULL;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet::Setup =====================================================

SetupTool_Linux::~SetupTool_Linux()
{
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aDebug
void SetupTool_Linux::UpdateInitRamFS()
{
    char lCmd[ 1024 ];

    sprintf( lCmd, "update-initramfs -u > " TMP_UPDATE_INIT_RAM_FS );

    if ( mDebug ) { printf( "system( \"%s\" );\n", lCmd ); }
    int lRet = system( lCmd );
    if ( 0 != lRet )
    {
        fprintf( stderr, "ERROR  %s  failed - %d\n", lCmd, lRet );
        fprintf( stderr, "See" TMP_UPDATE_INIT_RAM_FS "\n" );
    }
}

// ===== Info ===============================================================

const SetupTool_Linux::Info & SetupTool_Linux::Info_Get( unsigned int aIndex ) const
{
    Info_Map::const_iterator lIt = mInfos.begin();

    for ( unsigned int i = 0; i < aIndex; i ++ )
    {
        lIt ++;
    }

    assert( mInfos.end() != lIt );

    return lIt->second;
}

void SetupTool_Linux::Infos_Retrieve()
{
    mInfos.clear();

    PCI_Read();
    Mod_Read();
}

// ===== Mod ================================================================

// aName [---;R--] The name of the module
void SetupTool_Linux::Mod_BlackList_Disable( const char * aName )
{
    assert( NULL != aName );

    char lFileName[ 1024 ];

    sprintf( lFileName, ETC_MOD_PROBE_D "/blacklist-%s.conf", aName );

    if ( 0 == access( aName, R_OK | W_OK ) )
    {
        if ( mDebug ) { printf( "unlink( \"%s\" );\n", lFileName ); }
        int lRet = unlink( lFileName );
        if ( 0 != lRet )
        {
            fprintf( stderr, "ERROR  unlink( \"%s\" )  failed - %d\n", lFileName, lRet );
        }

        UpdateInitRamFS();
    }
}

// aName [---;R--] The name of the module to blacklist
void SetupTool_Linux::Mod_BlackList_Enable( const char * aName )
{
    assert( NULL != aName );

    char lFileName[ 1024 ];

    sprintf( lFileName, ETC_MOD_PROBE_D "/blacklist-%s.conf", aName );

    if ( mDebug ) { printf( "fopen( \"%s\", \"w\" );\n", lFileName ); }
    FILE * lFile = fopen( lFileName, "w" );
    if ( NULL == lFile )
    {
        fprintf( stderr, "ERROR  fopen( \"%s\", \"w\" )  failed\n", lFileName );
        return;
    }

    int lRet = fprintf( lFile, "blacklist %s\n", aName );
    assert( 0 < lRet );

    lRet = fclose( lFile );
    assert( 0 == lRet );

    UpdateInitRamFS();
}

// aName [---;R--]
void SetupTool_Linux::Mod_Probe_Disable( const char * aName )
{
    assert( NULL != aName );

    char lCmd[ 1024 ];

    sprintf( lCmd, "modprobe -r %s > " TMP_MOD_PROBE, aName );

    if ( mDebug ) { printf( "system( \"%s\" );\n", lCmd ); }
    int lRet = system( lCmd );
    if ( 0 != lRet )
    {
        fprintf( stderr, "ERROR  modprobe -r %s  failed - %d\n", aName, lRet );
        fprintf( stderr, "See " TMP_MOD_PROBE "\n" );
    }
}

// aName [---;R--]
void SetupTool_Linux::Mod_Probe_Enable( const char * aName )
{
    assert( NULL != aName );

    char lCmd[ 1024 ];

    sprintf( lCmd, "modprobe %s > " TMP_MOD_PROBE, aName );

    if ( mDebug ) { printf( "system( \"%s\" );\n", lCmd ); }
    int lRet = system( lCmd );
    if ( 0 != lRet )
    {
        fprintf( stderr, "ERROR  modprobe %s  failed - %d\n", aName, lRet );
        fprintf( stderr, "See " TMP_MOD_PROBE "\n" );
    }
}

void SetupTool_Linux::Mod_Read()
{
    char lCmd[ 1024 ];

    sprintf( lCmd, "lsmod > " TMP_LS_MOD );

    if ( mDebug ) { printf( "system( \"%s\" );\n", lCmd ); }
    int lRet = system( lCmd );
    if ( 0 != lRet )
    {
        fprintf( stderr, "ERROR  %s  failed - %d\n", lCmd, lRet );
        fprintf( stderr, "See " TMP_LS_MOD "\n" );
        return;
    }

    FILE * lFile = fopen( TMP_LS_MOD, "r" );
    assert( NULL != lFile );

    char lLine[ 1024 ];

    while ( NULL != fgets( lLine, sizeof( lLine ), lFile ) )
    {
        char lModule[ 1024 ];

        if ( ( 1 == sscanf(lLine, "%[^ \n\r\t]", lModule) ) && Mod_Known( lModule ) )
        {
            for ( SetupTool_Linux::Info_Map::iterator lIt = mInfos.begin(); lIt != mInfos.end(); lIt ++ )
            {
                const Module_Info * lMI = Mod_Find( lModule, static_cast< unsigned short >( lIt->second.mVendorId ), static_cast< unsigned short >( lIt->second.mDeviceId ) );
                if ( NULL != lMI )
                {
                    assert( NULL         != lMI->mName );
                    assert( MOD_TYPE_QTY >  lMI->mType );

                    lIt->second.mModule = lMI->mName;
                    lIt->second.mType   = lMI->mType;
                    break;
                }
            }
        }
    }

    fclose( lFile );
}

// ===== PCI ================================================================

void SetupTool_Linux::PCI_Read()
{
    char lCmd[ 1024 ];

    sprintf( lCmd, "lspci -n > " TMP_LS_PCI );

    if ( mDebug ) { printf( "system( \"%s\" )\n", lCmd ); }
    int lRet = system( lCmd );
    if ( 0 != lRet )
    {
        fprintf( stderr, "ERROR  %s  failed - %d\n", lCmd, lRet );
        fprintf( stderr, "See " TMP_LS_PCI "\n" );
        return;
    }

    FILE * lFile = fopen( TMP_LS_PCI, "r" );
    assert( NULL != lFile );

    char lLine[ 1024 ];

    while ( NULL != fgets( lLine, sizeof( lLine ), lFile ) )
    {
        unsigned int lA;
        unsigned int lB;
        unsigned int lC;
        unsigned int lD;
        Info         lInfo;

        memset( & lInfo, 0, sizeof( lInfo ) );

        if ( 6 == sscanf( lLine, "%x:%x.%x %x: %x:%x", & lA, & lB, & lC, & lD, & lInfo.mVendorId, & lInfo.mDeviceId ) )
        {
            const char * lName = GetAdapterName( lInfo.mVendorId, lInfo.mDeviceId );
            if ( NULL != lName )
            {
                Info_Map::iterator lIt = mInfos.find(lName);
                if ( mInfos.end() == lIt )
                {
                    lInfo.mCount = 1;
                    lInfo.mName  = lName;
                    lInfo.mType  = MOD_TYPE_NONE;

                    mInfos.insert( Info_Map::value_type( lName, lInfo ) );
                }
                else
                {
                    lIt->second.mCount ++;
                }
            }
        }
    }

    fclose( lFile );
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== Mod ================================================================

// aName [---;R--]
// aVendorId
// aDeviceId
const Module_Info * Mod_Find( const char * aName, unsigned short aVendorId, unsigned short aDeviceId )
{
    assert( NULL != aName );

    for ( unsigned int i = 0; i < MODULE_COUNT; i ++ )
    {
        const Module_Info & lMI = MODULES[ i ];

        assert( NULL != lMI.mName );

        if ( ( 0 == strcmp( lMI.mName, aName ) ) && ( lMI.mVendorId == aVendorId ) && ( lMI.mDeviceId == aDeviceId ) )
        {
            return ( & lMI );
        }
    }

    return NULL;
}

// aVendorId
// aDeviceId
const Module_Info & Mod_Find( unsigned short aVendorId, unsigned short aDeviceId )
{
    for ( unsigned int i = 0; i < MODULE_COUNT; i ++ )
    {
        const Module_Info & lMI = MODULES[ i ];

        if ( ( lMI.mVendorId == aVendorId ) && ( lMI.mDeviceId == aDeviceId ) && ( lMI.mType == SetupTool_Linux::MOD_TYPE_NET ) )
        {
            return lMI;
        }
    }

    assert( false );
}

// aName [---;R--]
bool Mod_Known( const char * aName )
{
    assert( NULL != aName );

    for ( unsigned int i = 0; i < MODULE_COUNT; i ++ )
    {
        assert( NULL != MODULES[ i ].mName );

        if ( 0 == strcmp( MODULES[ i ].mName, aName ) )
        {
            return true;
        }
    }

    return false;
}
