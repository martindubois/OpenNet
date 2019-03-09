
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/Processor.h
/// \brief      OpenNet::Processor

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    class UserBuffer;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the processor level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe d&eacute;finit l'interface au niveau du
    ///         processeur.
    /// \endcond
    class Processor
    {

    public:

        // TODO  OpenNet.Processor
        //       Low (Feature) - Ajouter un chemin de recherche additionnel
        //       pour les includes dans la configuration.

        // TODO  OpenNet.Processor
        //       Low (Feature) - Ajouter un parametre de configuration pour
        //       le chemin de recherche des include d'OpenNet.

        /// \cond en
        /// \brief  This structure contains the configuration of a Processor.
        /// \endcond
        /// \cond fr
        /// \brief  Cette structure contient la configuration d'un Processor.
        /// \endcond
        /// \todo   Document members
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
        /// \param  aIn  [---;R--] L'instance d'Info &agrave; afficher
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
        /// \brief   Retrieve the OpenCL context
        /// \return  This method returns a valid cl_context
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir le contexte OpenCL
        /// \return  Cette m&eacute;thode retourne une valeur cl_context
        ///          valide
        /// \endcond
        virtual void * GetContext() = 0;

        /// \cond en
        /// \brief   Retrieve the OpenCL device id
        /// \return  This method returns a valid cl_device_id or CUdevice
        /// \endcond
        /// \cond fr
        /// \brief   Obtenir l'indentificateur de device OpenCL
        /// \return  Cette m&eacute;thode retourne un cl_device_id ou
        ///          CUdevice valide
        /// \endcond
        virtual void * GetDevice() = 0;

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
        /// \retval Cette m&eacute;thode retourne l'adresse d'un espace
        ///         m&eacute;moire interne.
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
        /// \brief  Allocate a user buffer in the processor memory
        /// \param  aSize_byte  The size of the buffer
        /// \retval NULL   Error
        /// \retval Other  The UserBuffer instance
        /// \endcond
        /// \cond fr
        /// \brief  Allouer un espace m&eacute;moire utilisateur dans la
        ///         m&eacute;moire du Processor
        /// \param  aSize_byte  La taille de l'espace m&eacutemoire
        /// \retval NULL   Erreur
        /// \retval Other  L'instance de UserBuffer
        /// \endcond
        /// \sa     Kernel::SetStaticUserArgument, UserBuffer::Delete
        virtual UserBuffer * AllocateUserBuffer(unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual Status Display(FILE * aOut) const = 0;

    // Internal

        virtual ~Processor();

    protected:

        Processor();

    private:

        Processor(const Processor &);

        const Processor & operator = (const Processor &);

    };

}
