
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/Processor.cpp

#define __CLASS__     "Processor::"
#define __NAMESPACE__ "OpenNet::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== C ==================================================================
#include <stdint.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Processor.h>

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Status Processor::Display(const OpenNet::Processor::Info & aIn, FILE * aOut)
    {
        if (NULL == (&aIn)) { return STATUS_INVALID_REFERENCE        ; }
        if (NULL ==   aOut) { return STATUS_NOT_ALLOWED_NULL_ARGUMENT; }

        fprintf(aOut, "  Processor Information :\n");
        fprintf(aOut, "    Available                      = %s\n"              , aIn.mFlags.mAvailable         ? "true" : "false");
        fprintf(aOut, "    Compiler Available             = %s\n"              , aIn.mFlags.mCompilerAvailable ? "true" : "false");
        fprintf(aOut, "    Driver Version                 = %s\n"              , aIn.mDriverVersion              );
        fprintf(aOut, "    Endian Little                  = %s\n"              , aIn.mFlags.mEndianLittle      ? "true" : "false");
        fprintf(aOut, "    Global Mem - Cache - Line Size = %u bytes\n"        , aIn.mGlobalMemCacheLineSize_byte);
        // fprintf(aOut, "                       - Size      = %llu bytes\n"      , aIn.mGlobalMemCacheSize_byte    );
        fprintf(aOut, "                       - Type      = %u\n"              , aIn.mGlobalMemCacheType         );
        // fprintf(aOut, "               - Size              = %llu\n"            , aIn.mGlobalMemSize_byte         );
        // fprintf(aOut, "    Image - 2D Max - Height        = %llu\n"            , aIn.mImage2DMaxHeight           );
        // fprintf(aOut, "                   - Width         = %llu\n"            , aIn.mImage2DMaxWidth            );
        // fprintf(aOut, "          - 3D Max - Depth         = %llu\n"            , aIn.mImage3DMaxDepth            );
        // fprintf(aOut, "                   - Height        = %llu\n"            , aIn.mImage3DMaxHeight           );
        // fprintf(aOut, "                   - Width         = %llu\n"            , aIn.mImage3DMaxWidth            );
        fprintf(aOut, "    Image - Support                = %s\n"              , aIn.mFlags.mImageSupport      ? "true" : "false" );
        // fprintf(aOut, "    Local Mem - Size               = %llu bytes\n"      , aIn.mLocalMemSize_byte          );
        fprintf(aOut, "    Local Mem - Type               = %u\n"              , aIn.mLocalMemType               );
        fprintf(aOut, "    Max - Compute Units            = %u\n"              , aIn.mMaxComputeUnits            );
        fprintf(aOut, "        - Constant - Args          = %u\n"              , aIn.mMaxConstantArgs            );
        // fprintf(aOut, "                   - Buffer Size   = %llu bytes\n"      , aIn.mMaxConstantBufferSize_byte );
        // fprintf(aOut, "        - Mem Alloc Size           = %llu bytes\n"      , aIn.mMaxMemAllocSize_byte       );
        // fprintf(aOut, "        - Parameter Size           = %llu bytes\n"      , aIn.mMaxParameterSize_byte      );
        fprintf(aOut, "        - Read Image Args          = %u bytes\n"        , aIn.mMaxReadImageArgs           );
        fprintf(aOut, "        - Samplers                 = %u\n"              , aIn.mMaxSamplers                );
        // fprintf(aOut, "        - Work - Group Size        = %llu\n"            , aIn.mMaxWorkGroupSize           );
        // fprintf(aOut, "               - Item Sizes        = %llu, %llu, %llu\n", aIn.mMaxWorkItemSizes[0], aIn.mMaxWorkItemSizes[1], aIn.mMaxWorkItemSizes[2]);
        fprintf(aOut, "        - Write Image Args         = %u\n"              , aIn.mMaxWriteImageArgs          );
        fprintf(aOut, "    Mem Base Addr Align            = %u bits\n"         , aIn.mMemBaseAddrAlign_bit       );
        fprintf(aOut, "    Min Data Type Align Size       = %u bytes\n"        , aIn.mMinDataTypeAlignSize_byte  );
        fprintf(aOut, "    Name                           = %s\n"              , aIn.mName                       );
        fprintf(aOut, "    Preferred Vector Width - Char  = %u\n"              , aIn.mPreferredVectorWidthChar   );
        fprintf(aOut, "                           - Int   = %u\n"              , aIn.mPreferredVectorWidthInt    );
        fprintf(aOut, "                           - Long  = %u\n"              , aIn.mPreferredVectorWidthLong   );
        fprintf(aOut, "                           - Short = %u\n"              , aIn.mPreferredVectorWidthShort  );
        fprintf(aOut, "    Profile                        = %s\n"              , aIn.mProfile					 );
        fprintf(aOut, "    Vendor                         = %s\n"              , aIn.mVendor					 );
        fprintf(aOut, "           - Id                    = 0x%04x\n"          , aIn.mVendorId					 );
        fprintf(aOut, "    Version                        = %s\n"              , aIn.mVersion                    );

        return STATUS_OK;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    Processor::~Processor()
    {
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    Processor::Processor()
    {
    }

}
