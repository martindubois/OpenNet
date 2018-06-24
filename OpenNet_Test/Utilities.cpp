
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet_Test/Utilities.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== OpenNet_Test =======================================================
#include "Utilities.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

static const char * MASK_NAMES[UTL_MASK_QTY] =
{
    "EQUAL"         ,
    "IGNORE"        ,
    "ABOVE"         ,
    "ABOVE_OR_EQUAL",
    "BELOW"         ,
    "BELOW_OR_EQUAL",
    "DIFFERENT"     ,
};

// Macro
/////////////////////////////////////////////////////////////////////////////

#define VALIDATE(F)  lResult += Validate(aIn.F, aExpected.F, aMask.F, #F)

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static unsigned int Validate(unsigned int aIn, unsigned int aExpected, unsigned int aMask, const char * aName);
static unsigned int Validate(const unsigned int * aIn, const unsigned int * aExpected, const unsigned int * aMask, const char * aName);

// Functions
/////////////////////////////////////////////////////////////////////////////

unsigned int Utl_Validate(const OpenNet::Adapter::Stats & aIn, const OpenNet::Adapter::Stats & aExpected, const OpenNet::Adapter::Stats & aMask)
{
    assert(NULL != (&aIn      ));
    assert(NULL != (&aExpected));
    assert(NULL != (&aMask    ));

    unsigned int lResult = 0;

    VALIDATE(mDll.mBuffer_Allocated                     );
    VALIDATE(mDll.mBuffer_Released                      );
    VALIDATE(mDll.mPacket_Send                          );
    VALIDATE(mDll.mRun_Entry                            );
    VALIDATE(mDll.mRun_Exception                        );
    VALIDATE(mDll.mRun_Exit                             );
    VALIDATE(mDll.mRun_Iteration_Queue                  );
    VALIDATE(mDll.mRun_Iteration_Wait                   );
    VALIDATE(mDll.mRun_Loop_Exception                   );
    VALIDATE(mDll.mRun_Loop_UnexpectedException         );
    VALIDATE(mDll.mRun_Loop_Wait                        );
    VALIDATE(mDll.mRun_Queue                            );
    VALIDATE(mDll.mReserved0                            );
    VALIDATE(mDll.mRun_UnexpectedException              );
    VALIDATE(mDriver.mAdapter.mBuffers_Process          );
    VALIDATE(mDriver.mAdapter.mBuffer_InitHeader        );
    VALIDATE(mDriver.mAdapter.mBuffer_Queue             );
    VALIDATE(mDriver.mAdapter.mBuffer_Receive           );
    VALIDATE(mDriver.mAdapter.mBuffer_Send              );
    VALIDATE(mDriver.mAdapter.mBuffer_SendPackets       );
    VALIDATE(mDriver.mAdapter.mIoCtl                    );
    VALIDATE(mDriver.mAdapter.mIoCtl_Config_Get         );
    VALIDATE(mDriver.mAdapter.mIoCtl_Config_Set         );
    VALIDATE(mDriver.mAdapter.mIoCtl_Connect            );
    VALIDATE(mDriver.mAdapter.mIoCtl_Info_Get           );
    VALIDATE(mDriver.mAdapter.mIoCtl_Packet_Send        );
    VALIDATE(mDriver.mAdapter.mIoCtl_Start              );
    VALIDATE(mDriver.mAdapter.mIoCtl_State_Get          );
    VALIDATE(mDriver.mAdapter.mIoCtl_Stats_Get          );
    VALIDATE(mDriver.mAdapter.mIoCtl_Stop               );
    VALIDATE(mDriver.mAdapter.mReserved0                );
    VALIDATE(mDriver.mAdapter.mTx_Packet                );
    VALIDATE(mDriver.mAdapter_NoReset.mIoCtl_Last       );
    VALIDATE(mDriver.mAdapter_NoReset.mIoCtl_Last_Result);
    VALIDATE(mDriver.mAdapter_NoReset.mIoCtl_Stats_Reset);
    VALIDATE(mDriver.mAdapter_NoReset.mReserved0        );
    VALIDATE(mDriver.mHardware.mD0_Entry                );
    VALIDATE(mDriver.mHardware.mD0_Exit                 );
    VALIDATE(mDriver.mHardware.mInterrupt_Disable       );
    VALIDATE(mDriver.mHardware.mInterrupt_Enable        );
    VALIDATE(mDriver.mHardware.mInterrupt_Process       );
    VALIDATE(mDriver.mHardware.mInterrupt_Process2      );
    VALIDATE(mDriver.mHardware.mPacket_Receive          );
    VALIDATE(mDriver.mHardware.mPacket_Send             );
    VALIDATE(mDriver.mHardware.mReserved0               );
    VALIDATE(mDriver.mHardware.mRx_Packet               );
    VALIDATE(mDriver.mHardware.mSetConfig               );
    VALIDATE(mDriver.mHardware.mStats_Get               );
    VALIDATE(mDriver.mHardware.mTx_Packet               );
    VALIDATE(mDriver.mHardware_NoReset.mReserved0       );
    VALIDATE(mDriver.mHardware_NoReset.mStats_Reset     );

    return lResult;
}

void Utl_ValidateInit(OpenNet::Adapter::Stats * aExpected, OpenNet::Adapter::Stats * aMask)
{
    memset(aExpected, 0, sizeof(OpenNet::Adapter::Stats));
    memset(aMask    , 0, sizeof(OpenNet::Adapter::Stats));
}

// Static functions
/////////////////////////////////////////////////////////////////////////////

unsigned int Validate(unsigned int aIn, unsigned int aExpected, unsigned int aMask, const char * aName)
{
    assert(NULL != aName);

    unsigned int lResult;

    switch (aMask)
    {
    case UTL_MASK_ABOVE         : lResult = (aIn <= aExpected); break;
    case UTL_MASK_ABOVE_OR_EQUAL: lResult = (aIn <  aExpected); break;
    case UTL_MASK_BELOW         : lResult = (aIn >= aExpected); break;
    case UTL_MASK_BELOW_OR_EQUAL: lResult = (aIn >  aExpected); break;
    case UTL_MASK_DIFFERENT     : lResult = (aIn == aExpected); break;
    case UTL_MASK_EQUAL         : lResult = (aIn != aExpected); break;
    case UTL_MASK_IGNORE        : lResult =                 0 ; break;

    default: assert(false);
    }

    if (lResult)
    {
        printf("%s - In = %u, Expected = %u , Mask = %s\n", aName, aIn, aExpected, MASK_NAMES[aMask]);
    }

    return lResult;
}

unsigned int Validate(const unsigned int * aIn, const unsigned int * aExpected, const unsigned int * aMask, const char * aName)
{
    assert(NULL != aIn      );
    assert(NULL != aExpected);
    assert(NULL != aMask    );
    assert(NULL != aName    );

    return Validate(*aIn, *aExpected, *aMask, aName);
}