
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/PacketGenerator.cpp

#define __CLASS__ "PacketGenerator::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== OpenNet ============================================================
#include "Internal/PacketGenerator_Internal.h"

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             PacketGenerator_Internal contructor raise an exception
    PacketGenerator * PacketGenerator::Create()
    {
        PacketGenerator * lResult;

        try
        {
            // new ==> delete  See Delete
            lResult = new PacketGenerator_Internal();
        }
        catch (...)
        {
            lResult = NULL;
        }

        return lResult;
    }

    OpenNet::Status PacketGenerator::Display(const PacketGenerator::Config & aConfig, FILE * aOut)
    {
        if (NULL == (&aConfig)) { return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut    ) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  PacketGenerator::Config :\n");

        if (0 < aConfig.mAllowedIndexRepeat) { fprintf(aOut, "    Allowed Index Repeat = %u\n"                         , aConfig.mAllowedIndexRepeat); }
        else                                 { fprintf(aOut, "    Allowed Index Repeat = %u <== ERROR  Invalid value\n", aConfig.mAllowedIndexRepeat); }

        if (0.0 < aConfig.mBandwidth_MiB_s)  { fprintf(aOut, "    Bandwidth            = %f MiB/s\n"                           , aConfig.mBandwidth_MiB_s); }
        else                                 { fprintf(aOut, "    Bandwidth            = %f MiB/s\n <== ERROR  Invalid value\n", aConfig.mBandwidth_MiB_s); }

        fprintf(aOut, "    Destination Ethernet = %02x:%02x:%02x:%02x:%02x:%02x\n",
            aConfig.mDestinationEthernet.mAddress[0],
            aConfig.mDestinationEthernet.mAddress[1],
            aConfig.mDestinationEthernet.mAddress[2],
            aConfig.mDestinationEthernet.mAddress[3],
            aConfig.mDestinationEthernet.mAddress[4],
            aConfig.mDestinationEthernet.mAddress[5]);

        fprintf(aOut, "    Destination IPv4     = %u.%u.%u.%u\n",
            aConfig.mDestinationIPv4.mAddress[0],
            aConfig.mDestinationIPv4.mAddress[1],
            aConfig.mDestinationIPv4.mAddress[2],
            aConfig.mDestinationIPv4.mAddress[3]);

        fprintf(aOut, "    Destination Port  = 0x%04x\n"  , aConfig.mDestinationPort );
        fprintf(aOut, "    Ethernet Protocol = 0x%04x\n"  , aConfig.mEthernetProtocol);
        fprintf(aOut, "    Index Offset      = %u bytes\n", aConfig.mIndexOffset_byte);
        fprintf(aOut, "    IPv4 Protocol     = 0x%02x\n"  , aConfig.mIPv4Protocol    );
        fprintf(aOut, "    Packet Size       = %u bytes\n", aConfig.mPacketSize_byte );
        fprintf(aOut, "    Protocol          = %u\n"      , aConfig.mProtocol        );

        fprintf(aOut, "    Source IPv4       = %u.%u.%u.%u\n",
            aConfig.mSourceIPv4.mAddress[0],
            aConfig.mSourceIPv4.mAddress[1],
            aConfig.mSourceIPv4.mAddress[2],
            aConfig.mSourceIPv4.mAddress[3]);

        fprintf(aOut, "    Source Port       = 0x%04x\n", aConfig.mSourcePort);

        return STATUS_OK;
    }

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal destructor raise an exception
    void PacketGenerator::Delete()
    {
        try
        {
            // printf( __CLASS__ "~Delete - delete 0x%lx (this)\n", reinterpret_cast< uint64_t >( this ) );

            // new ==> delete  See Create
            delete this;
        }
        catch (...)
        {
            printf( __CLASS__ "~Delete - Exception\n" );
        }
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    PacketGenerator::PacketGenerator()
    {
    }

    PacketGenerator::~PacketGenerator()
    {
    }

}
