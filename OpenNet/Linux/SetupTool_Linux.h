
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Linux/SetupTool_Linux.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <map>
#include <string>
#include <vector>

// ===== OpenNet ============================================================
#include "../Internal/SetupTool_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class SetupTool_Linux : public SetupTool_Internal
{

public:

    typedef enum
    {
        MOD_TYPE_NONE,

        MOD_TYPE_NET     ,
        MOD_TYPE_OPEN_NET,

        MOD_TYPE_QTY
    }
    Module_Type;

    typedef struct
    {
        unsigned int mCount;

        unsigned int mVendorId;
        unsigned int mDeviceId;

        Module_Type  mType;

        const char * mName  ;
        const char * mModule;
    }
    Info;

    typedef std::map<std::string, Info> Info_Map;

    SetupTool_Linux(bool aDebug);

    // ===== OprnNet::SetupTool =============================================

    virtual OpenNet::Status Install  ();
    virtual OpenNet::Status Uninstall();

    virtual OpenNet::Status Interactif_ExecuteCommand(unsigned int aCommand);
    virtual unsigned int    Interactif_GetCommandCount();
    virtual const char    * Interactif_GetCommandText (unsigned int aCommand);

    virtual OpenNet::Status Wizard_ExecutePage       (unsigned int * aPage, unsigned int aButton);
    virtual unsigned int    Wizard_GetPageButtonCount(unsigned int   aPage);
    virtual const char    * Wizard_GetPageButtonText (unsigned int   aPage, unsigned int aButton);
    virtual unsigned int    Wizard_GetPageCount      ();
    virtual const char    * Wizard_GetPageText       (unsigned int   aPage);
    virtual const char    * Wizard_GetPageTitle      (unsigned int   aPage);

protected:

    // ===== OpenNet::SetupTool =============================================
    virtual ~SetupTool_Linux();

private:

    void UpdateInitRamFS();

    // ===== Info ===========================================================

    const Info & Info_Get( unsigned int aIndex ) const;

    void Infos_Retrieve();

    // ===== Mod ============================================================
    void Mod_BlackList_Disable( const char * aName );
    void Mod_BlackList_Enable ( const char * aName );
    void Mod_Probe_Disable    ( const char * aName );
    void Mod_Probe_Enable     ( const char * aName );
    void Mod_Read             ();

    // ===== PCI ============================================================
    void PCI_Read();

    Info_Map mInfos;

};
