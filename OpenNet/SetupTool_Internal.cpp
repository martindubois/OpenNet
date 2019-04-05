
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/SetupTool_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <memory.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>

// ===== OpenNet ============================================================
#include "FolderFinder.h"
#include "Utils.h"

#include "SetupTool_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet::SetupTool =================================================

const char * SetupTool_Internal::GetBinaryFolder() const
{
    assert(NULL != gFolderFinder);
    
    return gFolderFinder->GetBinaryFolder();
}

const char * SetupTool_Internal::GetDriverFolder() const
{
    assert(NULL != gFolderFinder);

    return gFolderFinder->GetDriverFolder();
}

const char * SetupTool_Internal::GetIncludeFolder() const
{
    assert(NULL != gFolderFinder);

    return gFolderFinder->GetIncludeFolder();
}

const char * SetupTool_Internal::GetInstallFolder() const
{
    assert(NULL != gFolderFinder);

    return gFolderFinder->GetInstallFolder();
}

const char * SetupTool_Internal::GetLibraryFolder() const
{
    assert(NULL != gFolderFinder);

    return gFolderFinder->GetLibraryFolder();
}

bool SetupTool_Internal::IsDebugEnabled() const
{
    return mDebug;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

SetupTool_Internal::SetupTool_Internal(bool aDebug) : mDebug(aDebug)
{
}

const char * SetupTool_Internal::GetAdapterName(unsigned int aVendorId, unsigned int aDeviceId) const
{
    switch (aVendorId)
    {
    case 0x8086 :
        switch (aDeviceId)
        {
        case 0x10c9: return "Intel Gigabit ET Dual Port Server Adapter (82576)";
        case 0x10fb: return "Intel Ethernet Server Adapter X520 (82599)"       ;
        }
        break;
    }

    return NULL;
}

// ===== OpenNet::SetupTool =================================================

SetupTool_Internal::~SetupTool_Internal()
{
}
