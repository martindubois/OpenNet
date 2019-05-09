
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Function_Forward.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>
#include <string.h>

// ===== Includes ===========================================================
#include <OpenNet/Adapter.h>

#include <OpenNet/Function_Forward.h>

// ===== OpenNet ============================================================
#include "SourceCode_Forward.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define EOL "\n"

static const char * CODE =
"OPEN_NET_FUNCTION_DECLARE( Function_Forward )"                               EOL
"{"                                                                           EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                 EOL
                                                                              EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | DESTINATIONS;"    EOL
                                                                              EOL
"    OPEN_NET_FUNCTION_END"                                                   EOL
"}"                                                                           EOL;

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Function_Forward::Function_Forward() : mDestinations(0)
    {
        SetFunctionName("Function_Forward");
    }

    Status Function_Forward::AddDestination(Adapter * aAdapter)
    {
        OpenNet::Status lResult = SourceCode_Forward_AddDestination(this, &mDestinations, aAdapter);
        if (OpenNet::STATUS_OK == lResult)
        {
            GenerateCode();
        }

        return lResult;
    }

    Status Function_Forward::RemoveDestination(Adapter * aAdapter)
    {
        OpenNet::Status lResult = SourceCode_Forward_RemoveDestination(this, &mDestinations, aAdapter);
        if (OpenNet::STATUS_OK == lResult)
        {
            GenerateCode();
        }

        return lResult;
    }

    // ===== Function =======================================================

    Status Function_Forward::SetFunctionName(const char * aFunctionName)
    {
        Status lResult = Function::SetFunctionName(aFunctionName);

        if (STATUS_OK == lResult)
        {
            lResult = ResetCode();
            assert((STATUS_OK == lResult) || (STATUS_CODE_NOT_SET == lResult));

            GenerateCode();
        }

        return lResult;
    }

    // ===== SourceCode =====================================================

    Function_Forward::~Function_Forward()
    {
    }

    Status Function_Forward::Display(FILE * aOut) const
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "Function_Forward :\n");
        fprintf(aOut, "  Destinations = 0x%08x\n", mDestinations);

        return Function::Display(aOut);
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Function_Forward::GenerateCode()
    {
        Status lStatus = SetCode(CODE, static_cast<unsigned int>(strlen(CODE)));
        assert(STATUS_OK == lStatus);
        (void)(lStatus);

        SourceCode_Forward_GenerateCode(this, mDestinations);

        unsigned int lRet = Edit_Replace("Function_Forward", GetFunctionName());
        assert(1 == lRet);
        (void)(lRet);
    }
}
