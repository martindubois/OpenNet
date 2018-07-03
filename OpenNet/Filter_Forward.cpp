
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Filter_Forward.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Adapter.h>

#include <OpenNet/Filter_Forward.h>

// Constants
/////////////////////////////////////////////////////////////////////////////

#define EOL "\n"

static const char * CODE_KERNEL =
"#include <OpenNetK/Kernel.h>"                                                EOL
                                                                              EOL
"OPEN_NET_KERNEL_DECLARE"                                                     EOL
"{"                                                                           EOL
"    OPEN_NET_KERNEL_BEGIN"                                                   EOL
                                                                              EOL
"    if ( OPEN_NET_PACKET_STATE_RX_COMPLETED == lPacketInfo->mPacketState )"  EOL
"    {"                                                                       EOL
"        lPacketInfo->mPacketState = OPEN_NET_PACKET_STATE_PX_COMPLETED;"     EOL
"        lPacketInfo->mToSendTo    = DESTINATIONS;"                           EOL
"    }"                                                                       EOL
                                                                              EOL
"    OPEN_NET_KERNEL_END"                                                     EOL
"}"                                                                           EOL;

static const char * CODE_SUB_KERNEL =
"OPEN_NET_SUB_KERNEL_DECLARE"                                                 EOL
"{"                                                                           EOL
"    OPEN_NET_KERNEL_BEGIN"                                                   EOL
                                                                              EOL
"    if ( OPEN_NET_PACKET_STATE_RX_COMPLETED == lPacketInfo->mPacketState )"  EOL
"    {"                                                                       EOL
"        lPacketInfo->mPacketState = OPEN_NET_PACKET_STATE_PX_COMPLETED;"     EOL
"        lPacketInfo->mToSendTo    = DESTINATIONS;"                           EOL
"    }"                                                                       EOL
                                                                              EOL
"    OPEN_NET_KERNEL_END"                                                     EOL
"}"                                                                           EOL;

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Filter_Forward::Filter_Forward() : mDestinations(0)
    {
        GenerateCode();
    }

    Status Filter_Forward::AddDestination(Adapter * aAdapter)
    {
        if (NULL == aAdapter) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        unsigned int lAdapterNo;

        Status lResult = aAdapter->GetAdapterNo(&lAdapterNo);
        if (STATUS_OK == lResult)
        {
            uint32_t lDestination = 1 << lAdapterNo;

            if (0 == (mDestinations & lDestination))
            {
                mDestinations |= lDestination;

                lResult = ResetCode();
                assert(STATUS_OK == lResult);

                GenerateCode();
            }
            else
            {
                lResult = STATUS_DESTINATION_ALREADY_SET;
            }
        }

        return lResult;
    }

    Status Filter_Forward::RemoveDestination(Adapter * aAdapter)
    {
        if (NULL == aAdapter)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        if (0 == mDestinations)
        {
            return STATUS_NO_DESTINATION_SET;
        }

        unsigned int lAdapterNo;

        Status lResult = aAdapter->GetAdapterNo(&lAdapterNo);
        if (STATUS_OK == lResult)
        {
            uint32_t lDestination = 1 << lAdapterNo;

            if (0 == (mDestinations & lDestination))
            {
                lResult = STATUS_DESTINATION_NOT_SET;
            }
            else
            {
                mDestinations &= ~lDestination;

                lResult = ResetCode();
                assert(STATUS_OK == lResult);

                if (0 != mDestinations)
                {
                    GenerateCode();
                }
            }
        }
    
        return lResult;
    }

    Status Filter_Forward::ResetDestinations()
    {
        if (0 == mDestinations)
        {
            return STATUS_NO_DESTINATION_SET;
        }

        mDestinations = 0;

        Status lResult = ResetCode();
        assert(STATUS_OK == lResult);

        GenerateCode();

        return lResult;
    }

    // ===== Filter =========================================================

    Filter_Forward::~Filter_Forward()
    {
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Filter_Forward::GenerateCode()
    {
        Status lStatus;

        switch (GetMode())
        {
        case MODE_KERNEL     : lStatus = SetCode(CODE_KERNEL    , static_cast<unsigned int>(strlen(CODE_KERNEL    ))); break;
        case MODE_SUB_KERNEL : lStatus = SetCode(CODE_SUB_KERNEL, static_cast<unsigned int>(strlen(CODE_SUB_KERNEL))); break;

        default: assert(false);
        }

        assert(STATUS_OK == lStatus);
        (void)(lStatus);

        char lDestinations[16];

        sprintf_s(lDestinations, "0x%08x", mDestinations);

        unsigned int lRet = Edit_Replace("DESTINATIONS", lDestinations);
        assert(1 == lRet);
    }

}
