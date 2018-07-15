
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Processor.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/StatisticsProvider.h>

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
    class Processor : public StatisticsProvider
    {

    public:

        /// \cond en
        /// \brief  This structure contains the configuration of a Processor.
        /// \endcond
        /// \cond fr
        /// \brief  Cette structure contient la configuration d'un Processor.
        /// \endcond
        typedef struct
        {
            struct
            {
                unsigned mProfilingEnabled : 1;

                unsigned mReserved0 : 31;
            }
            mFlags;

            unsigned char mReserved0[60];
        }
        Config;

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
        /// \brief  Display
        /// \param  aIn  [---;R--] The Info instance to Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aIn  [---;R--] L'instance d'Info a afficher
        /// \param  aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_REFERENCE
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        static OPEN_NET_PUBLIC Status Display(const Info & aIn, FILE * aOut);

        /// \cond en
        /// \brief  Retrieve configuration
        /// \param  aOut [---;-W-] The configuration
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la configuration
        /// \param  aOut [---;-W-] La configuration
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetConfig(Config * aOut) const = 0;

        /// \cond en
        /// \brief  Retrieve the Info
        /// \param  aOut [---;RW-] The Info instance
        /// \endcond
        /// \cond fr
        /// \brief  Retrouver le Info
        /// \param  aOut [---;RW-] L'instance d'Info
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        virtual Status GetInfo(Info * aOut) const = 0;

        /// \cond en
        /// \brief  Retrieve the instance's name
        /// \retval This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nom de l'instance
        /// \retval Cette methode retourne l'adresse d'un espace memoire
        ///         interne.
        /// \endcond
        virtual const char * GetName() const = 0;

        /// \cond en
        /// \brief  Modify the configuration
        /// \param  aConfig [---;-W-] The configuration
        /// \endcond
        /// \cond fr
        /// \brief  Changer la configuration
        /// \param  aConfig [---;-W-] La configuration
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_INVALID_REFERENCE
        virtual Status SetConfig(const Config & aConfig) = 0;

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

        Processor();

    private:

        Processor(const Processor &);

        const Processor & operator = (const Processor &);

    };

}
