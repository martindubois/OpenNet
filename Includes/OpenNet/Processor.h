
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Processor.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Status.h>

namespace OpenNet
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the processor level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit l'interface au niveau du processeur.
    /// \endcond
    class Processor
    {

    public:

        /// \cond en
        /// \brief  This structure contains the information about a
        ///         Processor.
        /// \endcond
        /// \cond fr
        /// \brief  Cette structure contient les information au sujet d'un
        ///         Processor.
        /// \endcond
        typedef struct
        {
            uint64_t mGlobalMemCacheSize_byte   ;
            uint64_t mGlobalMemSize_byte        ;
            uint64_t mImage2DMaxHeight          ;
            uint64_t mImage2DMaxWidth           ;
            uint64_t mImage3DMaxDepth           ;
            uint64_t mImage3DMaxHeight          ;
            uint64_t mImage3DMaxWidth           ;
            uint64_t mLocalMemSize_byte         ;
            uint64_t mMaxConstantBufferSize_byte;
            uint64_t mMaxMemAllocSize_byte      ;
            uint64_t mMaxParameterSize_byte     ;
            uint64_t mMaxWorkGroupSize          ;
            uint64_t mMaxWorkItemSizes       [3];

            uint8_t mReserved0[8];

            uint32_t mGlobalMemCacheType         ;
            uint32_t mGlobalMemCacheLineSize_byte;
            uint32_t mLocalMemType               ;
            uint32_t mMaxComputeUnits            ;
            uint32_t mMaxConstantArgs            ;
            uint32_t mMaxReadImageArgs           ;
            uint32_t mMaxSamplers                ;
            uint32_t mMaxWriteImageArgs          ;
            uint32_t mMemBaseAddrAlign_bit       ;
            uint32_t mMinDataTypeAlignSize_byte  ;
            uint32_t mPreferredVectorWidthChar   ;
            uint32_t mPreferredVectorWidthShort  ;
            uint32_t mPreferredVectorWidthInt    ;
            uint32_t mPreferredVectorWidthLong   ;
            uint32_t mVendorId                   ;

            uint8_t mReserved1[64];

            struct
            {
                unsigned mAvailable         : 1;
                unsigned mCompilerAvailable : 1;
                unsigned mImageSupport      : 1;
                unsigned mEndianLittle      : 1;

                unsigned mReserved0 : 28;
            }
            mFlags;

            char mDriverVersion[128];
            char mName         [128];
            char mProfile      [128];
            char mVendor       [128];
            char mVersion      [128];

            uint8_t mReserved2[128];

        }
        Info;

        /// \cond en
        /// \brief  Display the Info
        /// \param  aIn  [---;R--] The Info
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche le Info
        /// \param  aIn  [---;R--] Le Info
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_REFERENCE
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Info & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Retrieve the Info
        /// \param  aOut [---;RW-] The Info
        /// \endcond
        /// \cond fr
        /// \brief  Retrouver le Info
        /// \param  aOut [---;RW-] Le Info
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetInfo(Info * aOut) const = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status Display(FILE * aOut) const = 0;

    protected:

        /// \cond en
        /// \brief  Default constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur par defaut
        /// \endcond
        Processor();

    private:

        Processor(const Processor &);

        const Processor & operator = (const Processor &);

    };

}
