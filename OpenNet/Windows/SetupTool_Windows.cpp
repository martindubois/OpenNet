
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Windows/SetupTool_Windows.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== Windows ============================================================

#include <Windows.h>

#include <cfgmgr32.h>
#include <newdev.h>
#include <SetupAPI.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== Includes ===========================================================
#include <OpenNetK/Interface.h>

// ===== OpenNet/Windows ====================================================
#include "../FolderFinder.h"

#include "SetupTool_Windows.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define ONK_HARDWARE    "ONK_Hardware"
#define ONK_PRO1000_INF "ONK_Pro1000.inf"

static const GUID CLASS_NET      = { 0x4d36e972, 0xe325, 0x11ce, { 0xbf, 0xc1, 0x08, 0x00, 0x2b, 0xe1, 0x03, 0x18 } };
static const GUID CLASS_OPEN_NET = { 0x36576B2C, 0xE9F4, 0x4671, { 0xA8, 0xEF, 0x4B, 0xFB, 0x46, 0xCD, 0xD1, 0x22 } };

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

static void DisplayLastError(const char * aFunction);

// Public
/////////////////////////////////////////////////////////////////////////////

SetupTool_Windows::SetupTool_Windows(bool aDebug) : SetupTool_Internal(aDebug)
{
    RetrieveInfo();
}

// ===== OpenNet::SetupTool =================================================

OpenNet::Status SetupTool_Windows::Install()
{
    assert(NULL != gFolderFinder);

    const char * lDriverFolder = gFolderFinder->GetDriverFolder();
    assert(NULL != lDriverFolder);

    char  lFileName[1024];
    char  lFolder  [1024];
    DWORD lInfo_byte;

    sprintf_s(lFileName, "%s\\" ONK_HARDWARE "\\" ONK_PRO1000_INF, lDriverFolder);
    sprintf_s(lFolder  , "%s\\" ONK_HARDWARE                     , lDriverFolder);

    // SetupCopyOEMInfA ==> SetupUninstallOEMInfA  See Uninstall_Internal
    if (!SetupCopyOEMInf(lFileName, lFolder, SPOST_PATH, 0, NULL, 0, &lInfo_byte, NULL))
    {
        if (mDebug) { DisplayLastError("SetupCopyOEMInf"); }
        return OpenNet::STATUS_NOT_ADMINISTRATOR;
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status SetupTool_Windows::Uninstall()
{
    // Create a new device info list because this time we want the present
    // and the absent OpenNet device.
    HDEVINFO lDevIS = SetupDiGetClassDevs(&OPEN_NET_DRIVER_INTERFACE, NULL, NULL, DIGCF_DEVICEINTERFACE);
    if (INVALID_HANDLE_VALUE == lDevIS)
    {
        if (mDebug) { DisplayLastError("SetupDiGetClassDevs"); }
        return OpenNet::STATUS_INTERNAL_ERROR;
    }

    char lInfFileName[64];

    bool lUninstallInfFile = OpenNet_GetInfName(lInfFileName, sizeof(lInfFileName), lDevIS);

    BOOL lRetB;

    unsigned int lDevIndex = 0;
    for (;;)
    {
        SP_DEVINFO_DATA lDevID;

        if (!EnumDeviceInfo(lDevIS, lDevIndex, &lDevID))
        {
            break;
        }

        lDevIndex++;

        lRetB = SetupDiRemoveDevice(lDevIS, &lDevID);
        VerifyReturn(lRetB, "SetupDiRemoveDevice");
    }

    if (lUninstallInfFile)
    {
        // SetupCopyOEMInfA ==> SetupUninstallOEMInfA  See Install_Internal
        lRetB = SetupUninstallOEMInf(lInfFileName, 0, NULL);
        VerifyReturn(lRetB, "SetupUninstallOEMInf");
    }

    lRetB = SetupDiDestroyDeviceInfoList(lDevIS);
    VerifyReturn(lRetB, "SetupDiDestroyDeviceInfoList");

    ScanForHardwareChanges();

    return OpenNet::STATUS_OK;
}

OpenNet::Status SetupTool_Windows::Interactif_ExecuteCommand(unsigned int aCommand)
{
    unsigned int lCommand = aCommand;
    if (mNet_Infos.size() > lCommand)
    {
        return Net_Install(lCommand);
    }

    lCommand -= static_cast<unsigned int>(mNet_Infos.size());
    if (mOpenNet_Infos.size() > lCommand)
    {
        return OpenNet_Uninstall(lCommand);
    }

    return OpenNet::STATUS_INVALID_COMMAND_INDEX;
}

unsigned int SetupTool_Windows::Interactif_GetCommandCount()
{
    return static_cast<unsigned int>(mNet_Infos.size() + mOpenNet_Infos.size());
}

const char * SetupTool_Windows::Interactif_GetCommandText(unsigned int aCommand)
{
    unsigned int lCommand = aCommand;
    if (mNet_Infos.size() > lCommand)
    {
        const Info & lInfo = Net_GetInfo(aCommand);

        if (1 >= lInfo.mCount)
        {
            sprintf_s(mText, "Install OpenNet driver for\n    %s\n    (%s)", lInfo.mFriendlyName, lInfo.mLocation);
        }
        else
        {
            sprintf_s(mText, "Install OpenNet driver on the %u\n    %s", lInfo.mCount, lInfo.mFriendlyName);
        }
        return mText;
    }

    lCommand -= static_cast<unsigned int>(mNet_Infos.size());
    if (mOpenNet_Infos.size() > lCommand)
    {
        const Info & lInfo = mOpenNet_Infos[lCommand];

        sprintf_s(mText, "Uninstall OpenNet driver from\n    %s\n    (%s)", lInfo.mFriendlyName, lInfo.mLocation);
        return mText;
    }

    return NULL;
}

OpenNet::Status SetupTool_Windows::Wizard_ExecutePage(unsigned int * aPage, unsigned int aButton)
{
    assert(NULL != aPage);

    OpenNet::Status lResult;

    unsigned int lPage = (*aPage);
    if (mNet_Infos.size() > lPage)
    {
        switch (aButton)
        {
        case 0: lResult = Net_Install(lPage); break;
        case 1: lResult = OpenNet::STATUS_OK; (*aPage)++; break;

        default: lResult = OpenNet::STATUS_INVALID_BUTTON_INDEX;
        }

        return lResult;
    }

    lPage -= static_cast<unsigned int>(mNet_Infos.size());
    if (mOpenNet_Infos.size() > lPage)
    {
        switch (aButton)
        {
        case 0: lResult = OpenNet_Uninstall(lPage); (*aPage) = static_cast<unsigned int>(mNet_Infos.size()) + lPage; break;
        case 1: lResult = OpenNet::STATUS_OK      ; (*aPage)++; break;

        default: lResult = OpenNet::STATUS_INVALID_BUTTON_INDEX;
        }
        
        return lResult;
    }

    return OpenNet::STATUS_INVALID_PAGE_INDEX;
}

unsigned int SetupTool_Windows::Wizard_GetPageButtonCount(unsigned int aPage)
{
    if (static_cast<unsigned int>(mNet_Infos.size() + mOpenNet_Infos.size()) > aPage)
    {
        return 2;
    }

    return 0;
}

const char * SetupTool_Windows::Wizard_GetPageButtonText(unsigned int aPage, unsigned int aButton)
{
    unsigned int lPage = aPage;
    if (mNet_Infos.size() > lPage)
    {
        switch (aButton)
        {
        case 0: return "Install";
        case 1: return "Skip"   ;

        default: return NULL;
        }
    }

    lPage -= static_cast<unsigned int>(mNet_Infos.size());
    if (mOpenNet_Infos.size() > lPage)
    {
        switch (aButton)
        {
        case 0: return "Uninstall";
        case 1: return "Skip"     ;

        default: return NULL;
        }
    }

    return NULL;
}

unsigned int SetupTool_Windows::Wizard_GetPageCount()
{
    return static_cast<unsigned int>(mNet_Infos.size() + mOpenNet_Infos.size());
}

const char * SetupTool_Windows::Wizard_GetPageText(unsigned int aPage)
{
    unsigned int lPage = aPage;
    if (mNet_Infos.size() > lPage)
    {
        const Info & lInfo = Net_GetInfo(lPage);

        if (1 >= lInfo.mCount)
        {
            sprintf_s(mText,
                "The network adapter\n"
                "%s\n"
                "(%s)\n"
                "is compatible with OpenNet.\n"
                "\n"
                "It is currently configured to be used as a normal network\n"
                "adapter by the operating system.\n"
                "\n"
                "Installing the OpenNet driver would allow you to use it with\n"
                "OpenNet compatible softwares.\n"
                "\n"
                "Do you want to install the OpenNet driver?\n",
                lInfo.mFriendlyName, lInfo.mLocation);
        }
        else
        {
            sprintf_s(mText,
                "The %u network adapters\n"
                "%s\n"
                "are compatible with OpenNet.\n"
                "\n"
                "They are currently configured to be used as a normal network\n"
                "adapter by the operating system.\n"
                "\n"
                "Installing the OpenNet driver would allow you to use them with\n"
                "OpenNet compatible softwares.\n"
                "\n"
                "Do you want to install the OpenNet driver?\n",
                lInfo.mCount, lInfo.mFriendlyName);
        }

        return mText;
    }

    lPage -= static_cast<unsigned int>(mNet_Infos.size());
    if (mOpenNet_Infos.size() > lPage)
    {
        const Info & lInfo = mOpenNet_Infos[lPage];

        sprintf_s(mText,
            "The network adapter\n"
            "%s\n"
            "(%s)\n"
            "is compatible with OpenNet.\n"
            "\n"
            "It is currently configured to be used with OpenNet compatible\n"
            "softwares.\n"
            "\n"
            "Uninstalling the OpenNet driver would allow the operating\n"
            "system to use it as a normal network adapter.\n"
            "\n"
            "Do you want to uninstall the OpenNet driver?\n",
            lInfo.mFriendlyName, lInfo.mLocation);

        return mText;
    }

    return NULL;
}

const char * SetupTool_Windows::Wizard_GetPageTitle(unsigned int aPage)
{
    unsigned int lPage = aPage;
    if (mNet_Infos.size() > lPage)
    {
        return "Install OpenNet driver";
    }

    lPage -= static_cast<unsigned int>(mNet_Infos.size());
    if (mOpenNet_Infos.size() > lPage)
    {
        return "Uninstall OpenNet driver";
    }

    return NULL;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet::SetupTool =================================================

SetupTool_Windows::~SetupTool_Windows()
{
    ReleaseInfo();
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aDevIS
// aDevID [---;RW-]
void SetupTool_Windows::AllowExcludedDrivers(HDEVINFO aDevIS, SP_DEVINFO_DATA * aDevID)
{
    assert(INVALID_HANDLE_VALUE != aDevIS);
    assert(NULL                 != aDevID);

    SP_DEVINSTALL_PARAMS lParams;

    memset(&lParams, 0, sizeof(lParams));

    lParams.cbSize = sizeof(lParams);

    BOOL lRetB = SetupDiGetDeviceInstallParams(aDevIS, aDevID, &lParams);
    VerifyReturn(lRetB, "SetupDiGetDeviceInstallParams");

    lParams.FlagsEx |= DI_FLAGSEX_ALLOWEXCLUDEDDRVS;

    lRetB = SetupDiSetDeviceInstallParams(aDevIS, aDevID, &lParams);
    VerifyReturn(lRetB, "SetupDiSetDeviceInstallParams");
}

void SetupTool_Windows::RetrieveInfo()
{
    mNet_DevIS = SetupDiGetClassDevs(&CLASS_NET, NULL, NULL, DIGCF_PRESENT);
    assert(INVALID_HANDLE_VALUE != mNet_DevIS);

    mOpenNet_DevIS = SetupDiGetClassDevs(&OPEN_NET_DRIVER_INTERFACE, NULL, NULL, DIGCF_DEVICEINTERFACE | DIGCF_PRESENT);
    assert(INVALID_HANDLE_VALUE != mOpenNet_DevIS);

    Net_Count    ();
    OpenNet_Count();
}

void SetupTool_Windows::ReleaseInfo()
{
    BOOL lRetB;

    if (INVALID_HANDLE_VALUE != mOpenNet_DevIS)
    {
        lRetB = SetupDiDestroyDeviceInfoList(mOpenNet_DevIS);
        VerifyReturn(lRetB, "SetupDiDestroyDeviceInfoList");

        mOpenNet_DevIS = INVALID_HANDLE_VALUE;
        mOpenNet_Infos.clear();
    }

    if (INVALID_HANDLE_VALUE != mNet_DevIS)
    {
        lRetB = SetupDiDestroyDeviceInfoList(mNet_DevIS);
        VerifyReturn(lRetB, "SetupDiDestroyDeviceInfoList");

        mNet_DevIS = INVALID_HANDLE_VALUE;
        mNet_Infos.clear();
    }
}

// aDevIS           The device info set
// aDevID [---;R--] The device information
// aInfo  [---;-W-] The output buffer
void SetupTool_Windows::RetrievePciIdsAndLocation(HDEVINFO aDevIS, SP_DEVINFO_DATA * aDevID, SetupTool_Windows::Info * aInfo)
{
    assert(INVALID_HANDLE_VALUE != aDevIS);
    assert(NULL != aDevID);

    memset(aInfo, 0, sizeof(SetupTool_Windows::Info));

    aInfo->mCount = 1;

    GetDeviceRegistryProperty(aDevIS, aDevID, SPDRP_HARDWAREID, aInfo->mHardwareId, sizeof(aInfo->mHardwareId));

    if (mDebug) { printf("%s\n", aInfo->mHardwareId); }

    if (0 == strncmp("PCI\\VEN_", aInfo->mHardwareId, 8))
    {
        aInfo->mVendorId = strtoul(aInfo->mHardwareId + 8 , NULL, 16);
        aInfo->mDeviceId = strtoul(aInfo->mHardwareId + 17, NULL, 16);

        GetDeviceRegistryProperty(aDevIS, aDevID, SPDRP_LOCATION_INFORMATION, aInfo->mLocation, sizeof(aInfo->mLocation));

        if (mDebug) { printf("%s\n", aInfo->mLocation); }
    }
}

void SetupTool_Windows::ScanForHardwareChanges()
{
    DEVINST lRoot;

    CONFIGRET lCfgRet = CM_Locate_DevNode(&lRoot, NULL, 0);
    VerifyConfigRet(lCfgRet, "CM_LocateDevNode");

    lCfgRet = CM_Reenumerate_DevInst(lRoot, CM_REENUMERATE_SYNCHRONOUS);
    VerifyConfigRet(lCfgRet, "CM_Reenumerate_DevInst");
}

// aRet                See CR_...
// aFunction [---;R--]
void SetupTool_Windows::VerifyConfigRet(DWORD aRet, const char * aFunction)
{
    assert(NULL != aFunction);

    if (mDebug && (CR_SUCCESS != aRet))
    {
        printf("%s returned %u (0x%08x) ", aFunction, aRet, aRet);

        switch (aRet)
        {
        case CR_FAILURE: printf("CR_FAILURE"); break;
        }

        printf("\n");
    }

    assert(CR_SUCCESS == aRet);
}

// aRet
// aFunction [---;R--]
void SetupTool_Windows::VerifyReturn(BOOL aRet, const char * aFunction)
{
    assert(NULL != aFunction);

    if (mDebug && (!aRet))
    {
        DisplayLastError(aFunction);
    }

    assert(aRet);
}

// ===== Net ================================================================

void SetupTool_Windows::Net_Count()
{
    assert(INVALID_HANDLE_VALUE != mNet_DevIS);

    unsigned int lIndex = 0;

    for (;;)
    {
        SP_DEVINFO_DATA lDevID;

        if (!EnumDeviceInfo(mNet_DevIS, lIndex, &lDevID))
        {
            break;
        }

        Info lInfo;

        RetrievePciIdsAndLocation(mNet_DevIS, &lDevID, &lInfo);

        const char * lName = GetAdapterName(lInfo.mVendorId, lInfo.mDeviceId);
        if (NULL != lName)
        {
            Info_Map::iterator lIt = mNet_Infos.find(lInfo.mHardwareId);
            if (mNet_Infos.end() == lIt)
            {
                GetDeviceRegistryProperty(mNet_DevIS, &lDevID, SPDRP_FRIENDLYNAME, lInfo.mFriendlyName, sizeof(lInfo.mFriendlyName));

                lInfo.mIndex = lIndex;

                mNet_Infos.insert(Info_Map::value_type(lInfo.mHardwareId, lInfo));
            }
            else
            {
                lIt->second.mCount++;

                strcpy_s(lIt->second.mFriendlyName, lName);
            }
        }

        lIndex++;
    }
}

// aIndex
const SetupTool_Windows::Info & SetupTool_Windows::Net_GetInfo(unsigned int aIndex)
{
    Info_Map::iterator lIt = mNet_Infos.begin();

    for (unsigned int i = 0; i < aIndex; i++)
    {
        lIt++;
    }

    assert(mNet_Infos.end() != lIt);

    return lIt->second;
}

// aIndex  The index of the device or device group
OpenNet::Status SetupTool_Windows::Net_Install(unsigned int aIndex)
{
    const Info & lInfo = Net_GetInfo(aIndex);

    char lInfName[1024];
    BOOL lReboot;

    sprintf_s(lInfName, "%s\\" ONK_HARDWARE "\\" ONK_PRO1000_INF, gFolderFinder->GetDriverFolder());

    BOOL lRetB = UpdateDriverForPlugAndPlayDevices(NULL, lInfo.mHardwareId, lInfName, INSTALLFLAG_FORCE, &lReboot);
    VerifyReturn(lRetB, "UpdateDriverForPlugAndPlayDevices");

    if (lReboot)
    {
        return OpenNet::STATUS_REBOOT_REQUIRED;
    }

    ReleaseInfo ();
    RetrieveInfo();

    return OpenNet::STATUS_OK;
}

// ===== OpenNet ============================================================

void SetupTool_Windows::OpenNet_Count()
{
    assert(INVALID_HANDLE_VALUE != mOpenNet_DevIS);

    unsigned int lIndex = 0;

    for (;;)
    {
        SP_DEVINFO_DATA lDevID;

        if (!EnumDeviceInfo(mOpenNet_DevIS, lIndex, &lDevID))
        {
            break;
        }

        Info lInfo;

        RetrievePciIdsAndLocation(mOpenNet_DevIS, &lDevID, &lInfo);

        lInfo.mIndex = lIndex;

        const char * lName = GetAdapterName(lInfo.mVendorId, lInfo.mDeviceId);
        assert(NULL != lName);

        strcpy_s(lInfo.mFriendlyName, lName);

        mOpenNet_Infos.push_back(lInfo);

        lIndex++;
    }
}

// aDevIS           The device info set
// aDevID [---;R--] The device info
// aDrvID [---;-W-] The output buffer
void SetupTool_Windows::OpenNet_FindDriver(HDEVINFO aDevIS, SP_DEVINFO_DATA * aDevID, SP_DRVINFO_DATA * aDrvID)
{
    assert(INVALID_HANDLE_VALUE != aDevIS);
    assert(NULL                 != aDevID);
    assert(NULL                 != aDrvID);

    unsigned int lDrvIndex = 0;

    do
    {
        memset(aDrvID, 0, sizeof(SP_DRVINFO_DATA));

        aDrvID->cbSize = sizeof(SP_DRVINFO_DATA);

        BOOL lRetB = SetupDiEnumDriverInfo(aDevIS, aDevID, SPDIT_COMPATDRIVER, lDrvIndex, aDrvID);
        assert(lRetB);

        if (mDebug) { printf("%s\n", aDrvID->MfgName); }

        lDrvIndex++;
    }
    while (0 != strcmp("KMS", aDrvID->MfgName));
}

// aOut [---;-W-]
// aOutSize_byte
// aDevIS         The device info set
bool SetupTool_Windows::OpenNet_GetInfName(char * aOut, unsigned int aOutSize_byte, HDEVINFO aDevIS)
{
    assert(NULL                 != aOut         );
    assert(                   0 <  aOutSize_byte);
    assert(INVALID_HANDLE_VALUE != aDevIS       );

    SP_DEVINFO_DATA lDevID;

    if (!EnumDeviceInfo(aDevIS, 0, &lDevID))
    {
        return false;
    }

    AllowExcludedDrivers(aDevIS, &lDevID);

    BOOL lRetB = SetupDiBuildDriverInfoList(aDevIS, &lDevID, SPDIT_COMPATDRIVER);
    VerifyReturn(lRetB, "SetupDiBuildDriverInfoList");

    SP_DRVINFO_DATA lDrvID;

    OpenNet_FindDriver(aDevIS, &lDevID, &lDrvID);

    char                     lBuffer[4096];
    SP_DRVINFO_DETAIL_DATA * lDrvDetail = reinterpret_cast<SP_DRVINFO_DETAIL_DATA *>(lBuffer);
    DWORD                    lInfo_byte;

    memset(&lBuffer, 0, sizeof(lBuffer));

    lDrvDetail->cbSize = sizeof(SP_DRVINFO_DETAIL_DATA);

    lRetB = SetupDiGetDriverInfoDetail(aDevIS, &lDevID, &lDrvID, lDrvDetail, sizeof(lBuffer), &lInfo_byte);
    VerifyReturn(lRetB, "SetupDiGetDriverInfoDetail");

    const char * lPtr = strrchr(lDrvDetail->InfFileName, '\\');
    assert(NULL != lPtr);

    strncpy_s(aOut SIZE_INFO(aOutSize_byte), lPtr + 1, aOutSize_byte - 1);

    lRetB = SetupDiDestroyDriverInfoList(aDevIS, &lDevID, SPDIT_COMPATDRIVER);
    VerifyReturn(lRetB, "SetupDiDestroyDriverInfoList");

    return true;
}

// aIndex  The index of the device
OpenNet::Status SetupTool_Windows::OpenNet_Uninstall(unsigned int aIndex)
{
    assert(INVALID_HANDLE_VALUE != mOpenNet_DevIS);

    const Info & lInfo = mOpenNet_Infos[aIndex];

    SP_DEVINFO_DATA lDevID;

    bool lRet = EnumDeviceInfo(mOpenNet_DevIS, lInfo.mIndex, &lDevID);
    assert(lRet);

    BOOL lReboot;

    BOOL lRetB = DiUninstallDevice(NULL, mOpenNet_DevIS, &lDevID, 0, &lReboot);
    VerifyReturn(lRetB, "DiUninstallDevice");

    if (lReboot)
    {
        return OpenNet::STATUS_REBOOT_REQUIRED;
    }

    ScanForHardwareChanges();

    ReleaseInfo ();
    RetrieveInfo();

    return OpenNet::STATUS_OK;
}

// ===== SetupDi ============================================================

bool SetupTool_Windows::EnumDeviceInfo(HDEVINFO aDevIS, unsigned int aIndex, PSP_DEVINFO_DATA aDevID)
{
    assert(INVALID_HANDLE_VALUE != aDevIS);
    assert(NULL                 != aDevID);

    memset(aDevID, 0, sizeof(SP_DEVINFO_DATA));

    aDevID->cbSize = sizeof(SP_DEVINFO_DATA);

    if (SetupDiEnumDeviceInfo(aDevIS, aIndex, aDevID))
    {
        return true;
    }

    if (mDebug)
    {
        if (ERROR_NO_MORE_ITEMS != GetLastError())
        {
            DisplayLastError("SetupDiEnumDeviceInfo");
        }
    }

    return false;
}

void SetupTool_Windows::GetDeviceRegistryProperty(HDEVINFO aDevIS, PSP_DEVINFO_DATA aDevID, DWORD aProperty, void * aOut, unsigned int aOutSize_byte)
{
    assert(INVALID_HANDLE_VALUE != aDevIS       );
    assert(NULL                 != aDevID       );
    assert(NULL                 != aOut         );
    assert(                   0 <  aOutSize_byte);

    DWORD lInfo_byte;
    DWORD lType;

    BOOL lRetB = SetupDiGetDeviceRegistryProperty(aDevIS, aDevID, aProperty, &lType, reinterpret_cast<PBYTE>(aOut), aOutSize_byte, &lInfo_byte);
    VerifyReturn(lRetB, "SetupDiGetDeviceRegistryProperty");
    assert(aOutSize_byte >= lInfo_byte);
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

// aFunction [---;R--]
void DisplayLastError(const char * aFunction)
{
    assert(NULL != aFunction);

    DWORD lLastError = GetLastError();

    printf("%s failed and the last error is %u (0x%08x) ", aFunction, lLastError, lLastError);

    switch (lLastError)
    {
    case ERROR_CLASS_MISMATCH   : printf("ERROR_CLASS_MISMATCH"   ); break;
    case ERROR_INVALID_DATA     : printf("ERROR_INVALID_DATA"     ); break;
    case ERROR_INVALID_HANDLE   : printf("ERROR_INVALID_HANDLE"   ); break;
    case ERROR_INVALID_PARAMETER: printf("ERROR_INVALID_PARAMETER"); break;
    case ERROR_NO_MORE_ITEMS    : printf("ERROR_NO_MORE_ITEMS"    ); break;
    case ERROR_NO_SUCH_DEVINST  : printf("ERROR_NO_SUCH_DEVINST"  ); break;
    }

    printf("\n");
}
