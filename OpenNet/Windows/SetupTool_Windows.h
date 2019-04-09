
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Windows/SetupTool_Windows.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C++ ================================================================
#include <map>
#include <string>
#include <vector>

// ===== OpenNet/Windows ====================================================
#include "../Internal/SetupTool_Internal.h"

// Class
/////////////////////////////////////////////////////////////////////////////

class SetupTool_Windows : public SetupTool_Internal
{

public:

    typedef struct
    {
        unsigned int mCount;
        unsigned int mIndex;

        unsigned int mVendorId;
        unsigned int mDeviceId;

        char mFriendlyName[64];
        char mLocation    [64];

        char mHardwareId[1024];
    }
    Info;

    SetupTool_Windows(bool aDebug);

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
    virtual ~SetupTool_Windows();

private:

    typedef std::map<std::string, Info> Info_Map   ;
    typedef std::vector<Info>           Info_Vector;

    void AllowExcludedDrivers(HDEVINFO aDevIS, SP_DEVINFO_DATA * aDevID);

    void ReleaseInfo ();
    void RetrieveInfo();

    void RetrievePciIdsAndLocation(HDEVINFO aDevIS, SP_DEVINFO_DATA * aDevID, SetupTool_Windows::Info * aInfo);

    void ScanForHardwareChanges();

    void VerifyConfigRet(DWORD aRet, const char * aFunction);
    void VerifyReturn   (BOOL  aRet, const char * aFunction);

    // ===== Net ============================================================
    void            Net_Count  ();
    const Info    & Net_GetInfo(unsigned int aIndex);
    OpenNet::Status Net_Install(unsigned int aIndex);

    // ===== OpenNet ========================================================
    void            OpenNet_Count     ();
    void            OpenNet_FindDriver(HDEVINFO aDevIS, SP_DEVINFO_DATA * aDevID, SP_DRVINFO_DATA * aDrvID);
    bool            OpenNet_GetInfName(char * aOut, unsigned int aOutSize_byte, HDEVINFO aDevIS);
    OpenNet::Status OpenNet_Uninstall (unsigned int aIndex);

    // ===== SetupDi ========================================================
    bool EnumDeviceInfo           (HDEVINFO aDevIS, unsigned int aIndex, PSP_DEVINFO_DATA aDevID);
    void GetDeviceRegistryProperty(HDEVINFO aDevIS, PSP_DEVINFO_DATA aDevID, DWORD aProperty, void * aOut, unsigned int aOutSize_byte);

    HDEVINFO mNet_DevIS;
    Info_Map mNet_Infos;

    HDEVINFO    mOpenNet_DevIS;
    Info_Vector mOpenNet_Infos;

};
