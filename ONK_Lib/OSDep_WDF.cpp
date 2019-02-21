
// Auhtor     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Lib/OSDep_WDF.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_WDF.h>

// ===== ONK_Lib ============================================================
#include "OSDep_WDF.h"

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

// ===== OSDep ==============================================================

static void * AllocateMemory(unsigned int aSize_byte);
static void   FreeMemory(void * aMemory);

static uint64_t GetTimeStamp();

static void   LockSpinlock(void * aLock);
static void   UnlockSpinlock(void * aLock);

static void * MapBuffer(void * aContext, uint64_t * aBuffer_PA, uint64_t aBuffer_DA, unsigned int aSize_byte, uint64_t aMarker_PA, volatile void * * aMarker_MA );
static void   UnmapBuffer(void * aContext, void * aBuffer_MA, unsigned int aSize_byte, volatile void * aMarker_MA);

static void * MapSharedMemory(void * aContext, void * aShared_UA, unsigned int aSize_byte);
static void   UnmapSharedMemory(void * aContext);

// Function
/////////////////////////////////////////////////////////////////////////////

// aOSDep   [---;-W-;
// aContext [-KO;---]
void OSDep_Init(OpenNetK_OSDep * aOSDep, void * aContext)
{
    ASSERT(NULL != aOSDep);

    memset(aOSDep, 0, sizeof(OpenNetK_OSDep));

    aOSDep->mContext = aContext;

    aOSDep->AllocateMemory    = AllocateMemory   ;
    aOSDep->FreeMemory        = FreeMemory       ;
    aOSDep->GetTimeStamp      = GetTimeStamp     ;
    aOSDep->LockSpinlock      = LockSpinlock     ;
    aOSDep->MapBuffer         = MapBuffer        ;
    aOSDep->MapSharedMemory   = MapSharedMemory  ;
    aOSDep->UnlockSpinlock    = UnlockSpinlock   ;
    aOSDep->UnmapBuffer       = UnmapBuffer      ;
    aOSDep->UnmapSharedMemory = UnmapSharedMemory;
};

// Static functions
/////////////////////////////////////////////////////////////////////////////

// ===== OSDep ==============================================================

void * AllocateMemory(unsigned int aSize_byte)
{
    ASSERT(0 < aSize_byte);

    return ExAllocatePoolWithTag(NonPagedPool, aSize_byte, TAG);
}

void FreeMemory(void * aMemory)
{
    ASSERT(NULL != aMemory);

    ExFreePoolWithTag(aMemory, TAG);
}

uint64_t GetTimeStamp()
{
    LARGE_INTEGER lResult;

    KeQuerySystemTimePrecise(&lResult);

    return (lResult.QuadPart / 10); // 100 ns ==> us
}

void LockSpinlock(void * aLock)
{
    ASSERT(NULL != aLock);

    WdfSpinLockAcquire(reinterpret_cast<WDFSPINLOCK>(aLock));
}

void UnlockSpinlock(void * aLock)
{
    ASSERT(NULL != aLock);

    WdfSpinLockRelease(reinterpret_cast<WDFSPINLOCK>(aLock));
}

void * MapBuffer(void * aContext, uint64_t * aBuffer_PA, uint64_t aBuffer_DA, unsigned int aSize_byte, uint64_t aMarker_PA, volatile void * * aMarker_MA)
{
    ASSERT(NULL != aBuffer_PA);
    ASSERT(   0 <  aSize_byte);
    ASSERT(NULL != aMarker_MA);

    (void)(aContext  );
    (void)(aBuffer_DA);

    PHYSICAL_ADDRESS lPA;

    lPA.QuadPart = *aBuffer_PA;

    void * lResult = MmMapIoSpace(lPA, aSize_byte, MmNonCached);
    ASSERT(NULL != lResult);

    lPA.QuadPart = aMarker_PA;

    ( * aMarker_MA ) = reinterpret_cast<uint32_t *>(MmMapIoSpace(lPA, PAGE_SIZE, MmNonCached));
    ASSERT(NULL != (*aMarker_MA));

    return lResult;
}

void UnmapBuffer(void * aContext, void * aBuffer_MA, unsigned int aSize_byte, volatile void * aMarker_MA)
{
    ASSERT(NULL != aBuffer_MA);
    ASSERT(   0 <  aSize_byte);
    ASSERT(NULL != aMarker_MA);

    (void)(aContext);

    MmUnmapIoSpace((PVOID)( aMarker_MA ), PAGE_SIZE ); // volatile_cast
    MmUnmapIoSpace(         aBuffer_MA  , aSize_byte);
}

void * MapSharedMemory(void * aContext, void * aShared_UA, unsigned int aSize_byte)
{
    ASSERT(NULL != aShared_UA);
    ASSERT(   0 <  aSize_byte);

    (void)(aContext  );
    (void)(aSize_byte);

    // Because the mapping must be accomplished into the user thread, we do
    // not do it here. It has been done before, and the aShared_VA variable
    // already contains the mapped address.
    return aShared_UA;
}

void UnmapSharedMemory(void * aContext)
{
    ASSERT(NULL != aContext);

    OpenNetK::Adapter_WDF * lThis = reinterpret_cast<OpenNetK::Adapter_WDF *>(aContext);

    lThis->SharedMemory_Release();
}
