
// Author     KMS - Martin Dubois, P.Eng.
// Copyright  (C) 2018-2020 KMS. All rights reserved
// Product    OpenNet
// File       OpenNet/Kernel_Forward.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>
#include <stdio.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Adapter.h>

#include <OpenNet/Kernel_Forward.h>

// ===== OpenNet ============================================================
#include "SourceCode_Forward.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define EOL "\n"

static const char * CODE =
"#include <OpenNetK/Kernel.h>"                                                EOL
                                                                              EOL
"OPEN_NET_KERNEL_DECLARE"                                                     EOL
"{"                                                                           EOL
"    OPEN_NET_KERNEL_BEGIN"                                                   EOL
                                                                              EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | DESTINATIONS;"    EOL
                                                                              EOL
"    OPEN_NET_KERNEL_END"                                                     EOL
"}"                                                                           EOL;

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Kernel_Forward::Kernel_Forward() : mDestinations(0)
    {
        GenerateCode();
    }

    Status Kernel_Forward::AddDestination(Adapter * aAdapter)
    {
        Status lResult = SourceCode_Forward_AddDestination(this, &mDestinations, aAdapter);
        if (STATUS_OK == lResult)
        {
            GenerateCode();
        }

        return lResult;
    }

    Status Kernel_Forward::RemoveDestination(Adapter * aAdapter)
    {
        Status lResult = SourceCode_Forward_RemoveDestination(this, &mDestinations, aAdapter);
        if (STATUS_OK == lResult)
        {
            GenerateCode();
        }

        return lResult;
    }

    Status Kernel_Forward::ResetDestinations()
    {
        Status lResult = SourceCode_Forward_ResetDestinations(this, &mDestinations);
        if (OpenNet::STATUS_OK == lResult)
        {
            GenerateCode();
        }

        return lResult;
    }

    // ===== SourceCode =====================================================

    Kernel_Forward::~Kernel_Forward()
    {
    }

    Status Kernel_Forward::Display(FILE * aOut) const
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "Kernel_Forward :\n");
        fprintf(aOut, "  Destinations = 0x%08x\n", mDestinations);

        return Kernel::Display(aOut);
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Kernel_Forward::GenerateCode()
    {
        Status lStatus = SetCode(CODE, static_cast<unsigned int>(strlen(CODE)));
        assert(STATUS_OK == lStatus);
        (void)(lStatus);

        SourceCode_Forward_GenerateCode(this, mDestinations);
    }

}
