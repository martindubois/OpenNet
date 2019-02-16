
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       ONK_Pro1000/NvBuffer.h

#pragma once

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{

// private:

    uint64_t                   mAddress_DA;
    void                     * mAddress_MA;
    nvidia_p2p_dma_mapping_t * mMapping   ;
    nvidia_p2p_page_table_t  * mPageTable ;

}
NvBuffer;

// Functions
/////////////////////////////////////////////////////////////////////////////

extern uint64_t NvBuffer_GetPhysicalAddress( NvBuffer * aThis );
extern void   * NvBuffer_GetMappedAddress  ( NvBuffer * aThis );
extern int      NvBuffer_Is                ( NvBuffer * aThis, void * aAddress_MA );
extern int      NvBuffer_IsAvailable       ( NvBuffer * aThis );
extern int      NvBuffer_Map               ( NvBuffer * aThis, struct pci_dev * aPciDev, uint64_t aAddress_DA, unsigned int aSize_byte, void ( * aFreeCallback )( void * ), void * aData );
extern void     NvBuffer_Unmap             ( NvBuffer * aThis, struct pci_dev * aPciDev, bool aPutPage );
