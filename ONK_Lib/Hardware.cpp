
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     ONK_Lib/Hardware.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== WDM ================================================================

#define INITGUID

#include <ntddk.h>

// ===== WDF ================================================================
#include <wdf.h>

// ===== Includes ===========================================================
#include <OpenNetK/StdInt.h>

#include <OpenNetK/Adapter.h>
#include <OpenNetK/Interface.h>

#include <OpenNetK/Hardware.h>

// ===== Common =============================================================
#include "../Common/Version.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define SIZE_64_KB (64 * 1024)

namespace OpenNetK
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    void * Hardware::operator new(size_t aSize_byte, void * aAddress)
    {
        ASSERT(sizeof(Hardware) <= aSize_byte);
        ASSERT(NULL             != aAddress  );

        (void)(aSize_byte);

        return aAddress;
    }

    unsigned int Hardware::GetPacketSize() const
    {
        return mConfig.mPacketSize_byte;
    }

    void Hardware::GetState(OpenNet_State * aState)
    {
        ASSERT(NULL != aState);

        (void)(aState);
    }

    void Hardware::ResetMemory()
    {
    }

    void Hardware::SetAdapter(Adapter * aAdapter)
    {
        ASSERT(NULL != aAdapter);

        ASSERT(NULL == mAdapter);

        mAdapter = aAdapter;
    }

    // NOT TESTED  ONK_Lib.Hardware
    //             The SetCommonBuffer is only there to fill the virtual
    //             table entry when the driver does not need a common buffer.
    void Hardware::SetCommonBuffer(uint64_t aLogicalAddress, volatile void * aVirtualAddress)
    {
        ASSERT(NULL != aVirtualAddress);

        (void)(aLogicalAddress);
        (void)(aVirtualAddress);

        ASSERT(false);
    }

    void Hardware::SetConfig(const OpenNet_Config & aConfig)
    {
        ASSERT(NULL != (&aConfig));

        memcpy(&mConfig, &aConfig, sizeof(mConfig));
    }

    bool Hardware::SetMemory(unsigned int aIndex, volatile void * aVirtual, unsigned int aSize_byte)
    {
        ASSERT(NULL != aVirtual  );
        ASSERT(   0 <  aSize_byte);

        (void)(aVirtual  );
        (void)(aIndex    );
        (void)(aSize_byte);

        return true;
    }

    bool Hardware::D0_Entry()
    {
        return true;
    }

    bool Hardware::D0_Exit()
    {
        return true;
    }

    void Hardware::Interrupt_Disable()
    {
    }

    void Hardware::Interrupt_Enable()
    {
    }

    // NOT TESTED  ONK_Lib.Hardware
    //             The Interrupt_Process is only there to fill the virtual
    //             table entry when the driver does not need interrupt.
    bool Hardware::Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing)
    {
        ASSERT(NULL != aNeedMoreProcessing);

        (void)(aMessageId         );
        (void)(aNeedMoreProcessing);

        ASSERT(false);

        return false;
    }

    void Hardware::Interrupt_Process2()
    {
        ASSERT(NULL != mAdapter);

        mAdapter->Buffers_Process();
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    unsigned int Hardware::GetCommonBufferSize() const
    {
        return mInfo.mCommonBufferSize_byte;
    }

    void Hardware::GetConfig(OpenNet_Config * aConfig)
    {
        ASSERT(NULL != aConfig);

        memcpy(aConfig, &mConfig, sizeof(mConfig));
    }

    void Hardware::GetInfo(OpenNet_Info * aInfo)
    {
        ASSERT(NULL != aInfo);

        memcpy(aInfo, &mInfo, sizeof(mInfo));
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    void Hardware::Skip64KByteBoundary(uint64_t * aLogical, volatile uint8_t ** aVirtual, unsigned int aSize_byte, uint64_t * aOutLogical, volatile uint8_t ** aOutVirtual)
    {
        ASSERT(NULL       !=   aLogical   );
        ASSERT(NULL       !=   aVirtual   );
        ASSERT(NULL       != (*aVirtual  ));
        ASSERT(         0 <    aSize_byte );
        ASSERT(SIZE_64_KB >    aSize_byte );
        ASSERT(NULL       !=   aOutLogical);
        ASSERT(NULL       !=   aOutVirtual);

        uint64_t lEnd = (*aLogical) + aSize_byte - 1;

        if (((*aLogical) & SIZE_64_KB) == (lEnd & SIZE_64_KB))
        {
            (*aOutLogical) = (*aLogical);
            (*aOutVirtual) = (*aVirtual);
        }
        else
        {
            uint64_t lOffset_byte = SIZE_64_KB - ((*aLogical) % SIZE_64_KB);

            (*aOutLogical) = (*aLogical) + lOffset_byte;
            (*aOutVirtual) = (*aVirtual) + lOffset_byte;
        }

        (*aLogical) = (*aOutLogical) + aSize_byte;
        (*aVirtual) = (*aOutVirtual) + aSize_byte;
    }

    Hardware::Hardware() : mAdapter(NULL)
    {
        memset(&mConfig, 0, sizeof(mConfig));
        memset(&mInfo  , 0, sizeof(mInfo  ));

        mConfig.mMode            = OPEN_NET_MODE_NORMAL         ;
        mConfig.mPacketSize_byte = OPEN_NET_PACKET_SIZE_MAX_byte;

        mInfo.mAdapterType     = OPEN_NET_ADAPTER_TYPE_ETHERNET;
        mInfo.mPacketSize_byte = OPEN_NET_PACKET_SIZE_MAX_byte ;

        mInfo.mVersion_Driver.mMajor         = VERSION_MAJOR;
        mInfo.mVersion_Driver.mMinor         = VERSION_MINOR;
        mInfo.mVersion_Driver.mBuild         = VERSION_BUILD;
        mInfo.mVersion_Driver.mCompatibility = VERSION_COMPATIBILITY;
    
        mInfo.mVersion_ONK_Lib.mMajor         = VERSION_MAJOR;
        mInfo.mVersion_ONK_Lib.mMinor         = VERSION_MINOR;
        mInfo.mVersion_ONK_Lib.mBuild         = VERSION_BUILD;
        mInfo.mVersion_ONK_Lib.mCompatibility = VERSION_COMPATIBILITY;
    
        strcpy(mInfo.mVersion_Driver .mType   , VERSION_TYPE);
        strcpy(mInfo.mVersion_ONK_Lib.mComment, "ONK_Lib"   );
        strcpy(mInfo.mVersion_ONK_Lib.mType   , VERSION_TYPE);
    }

    Adapter * Hardware::GetAdapter()
    {
        ASSERT(NULL != mAdapter);

        return mAdapter;
    }

}
