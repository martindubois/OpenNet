
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

static const char * CODE =
"#include <OpenNetK/Kernel.h>"                                           EOL
                                                                         EOL
"OPEN_NET_KERNEL_DECLARE"                                                EOL
"{"                                                                      EOL
"    OPEN_NET_KERNEL_BEGIN"                                              EOL
                                                                         EOL
"    if ( OPEN_NET_PACKET_STATE_RECEIVED == lPacketInfo->mPacketState )" EOL
"    {"                                                                  EOL
"        lPacketInfo->mPacketState = OPEN_NET_PACKET_STATE_PROCESSED;"   EOL
"        lPacketInfo->mToSendTo    = DESTINATIONS;"                      EOL
"    }"                                                                  EOL
                                                                         EOL
"    OPEN_NET_KERNEL_END"                                                EOL
"}"                                                                      EOL;

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Filter_Forward::Filter_Forward()
    {
    }

    Status Filter_Forward::AddDestination(Adapter * aAdapter)
    {
        if (NULL == aAdapter) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        // TODO  Test

        unsigned int lAdapterNo;

        Status lResult = aAdapter->GetAdapterNo(&lAdapterNo);
        if (STATUS_OK == lResult)
        {
            uint32_t lDestination = 1 << lAdapterNo;

            if (0 == mDestinations)
            {
                mDestinations = lDestination;

                GenerateCode();
            }
            else
            {
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
        }

        return lResult;
    }

    Status Filter_Forward::RemoveDestination(Adapter * aAdapter)
    {
        if (NULL == aAdapter)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        // TODO  Test

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

        // TODO  Test

        mDestinations = 0;

        Status lResult = ResetCode();
        assert(STATUS_OK == lResult);

        return lResult;
    }

    // ===== Filter =========================================================

    Filter_Forward::~Filter_Forward()
    {
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // TODO  Test
    void Filter_Forward::GenerateCode()
    {
        assert(0 != mDestinations);

        Status lStatus = SetCode(CODE, static_cast<unsigned int>(strlen(CODE)));
        assert(STATUS_OK == lStatus);

        char lDestinations[16];

        sprintf_s(lDestinations, "0x%08x", mDestinations);

        unsigned int lRet = Edit_Replace("DESTINATIONS", lDestinations);
        assert(1 == lRet);
    }

}
