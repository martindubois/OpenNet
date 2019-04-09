
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/SetupTool_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/SetupTool.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class SetupTool_Internal : public OpenNet::SetupTool
{

public:

    // ===== OpenNet::SetupTool =============================================

    virtual const char * GetBinaryFolder () const;
    virtual const char * GetIncludeFolder() const;
    virtual const char * GetInstallFolder() const;
    virtual const char * GetLibraryFolder() const;

    virtual bool IsDebugEnabled() const;

protected:

    SetupTool_Internal(bool aDebug);

    const char * GetAdapterName(unsigned int aVendorId, unsigned int aDeviceId) const;

    // ===== OpenNet::SetupTool =============================================
    virtual ~SetupTool_Internal();

    bool mDebug;
    char mText[4096];

};
