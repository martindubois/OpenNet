
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Tool/PacketSender.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Common =============================================================
#include "../Common/Constants.h"

// ===== OpenNet_Tool =======================================================
#include "PacketSender.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const unsigned char PACKET[PACKET_SIZE_MAX_byte] =
{
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x88, 0x88
};

// Static function declaration
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================
static DWORD WINAPI Run(LPVOID aParameter);

// Public
/////////////////////////////////////////////////////////////////////////////

PacketSender::PacketSender(OpenNet::Adapter * aAdapter, unsigned int aPacketSize_byte, unsigned int aPacketQty)
    : mAdapter        (aAdapter        )
    , mPacketQty      (aPacketQty      )
    , mPacketSize_byte(aPacketSize_byte)
    , mState          (STATE_INIT      )
    , mThread         (NULL            )
{
    assert(NULL != aAdapter        );
    assert(   0 <  aPacketSize_byte);
    assert(   0 <  aPacketQty      );
}

void PacketSender::Start()
{
    mState = STATE_RUNNING;

    mThread = CreateThread(NULL, 0, ::Run, this, 0, &mThreadId);
}

void PacketSender::Stop()
{
    assert(   0 <  mDelay_ms);
    assert(NULL != mThread  );

    mState = STATE_STOPPING;

    DWORD lRet = WaitForSingleObject(mThread, 1500);
    assert(WAIT_OBJECT_0 == lRet);
    (void)(lRet);

    BOOL lRetB = CloseHandle(mThread);
    assert(lRetB);
    (void)(lRetB);

    mThread = NULL;
}

// Internal
/////////////////////////////////////////////////////////////////////////////

unsigned int PacketSender::Run()
{
    assert(0 < mDelay_ms);

    while (STATE_RUNNING == mState)
    {
        Sleep(15);

        for (unsigned int i = 0; i < mPacketQty; i++)
        {
            OpenNet::Status lStatus = mAdapter->Packet_Send(PACKET, mPacketSize_byte);
            assert(OpenMet::STATUS_OK == lStatus);
        }
    }

    return 0;
}

// Static function
/////////////////////////////////////////////////////////////////////////////

// ===== Entry point ========================================================

DWORD WINAPI Run(LPVOID aParameter)
{
    PacketSender * lThis = reinterpret_cast<PacketSender *>(aParameter);

    return lThis->Run();
}
