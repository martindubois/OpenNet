
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/UserBuffer.h
/// \brief      OpenNet::UserBuffer

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
    /// \brief  The UserBuffer class
    /// \endcond
    /// \cond fr
    /// \brief  La classe UserBuffer
    /// \endcond
    class UserBuffer
    {

    public:

        /// \cond en
        /// \brief  Initialise the buffer to 0
        /// \endcond
        /// \cond fr
        /// \brief  Initialiser l'espace m&eacute; &agrave; 0
        /// \endcond
        virtual OpenNet::Status Clear() = 0;

        /// \cond en
        /// \brief  Read data from a UserBuffer
        /// \param  aOffset_byte  Where to start to read in the UserBuffer
        /// \param  aOut          Where to put read data
        /// \param  aSize_byte    Size of data to read
        /// \endcond
        /// \cond fr
        /// \brief  Lire des donn&eacute;es d'un UserBuffer
        /// \param  aOffsetByte  O&ugrave; d&eacute;buter &agrave; lire dans
        ///                      le UserBuffer
        /// \param  aOut         O&ugrave; placer les donn&eacute;es lues
        /// \param  aSize_byte   La taille des donn&eacute;es &agrave; lire
        /// \endcond
        /// \retval STATUS_OK
        virtual OpenNet::Status Read(unsigned int aOffset_byte, void * aOut, unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  Write data to a UserBuffer
        /// \param  aOffset_byte  Where to start to write in the UserBuffer
        /// \param  aOut          Where to get the data to write
        /// \param  aSize_byte    Size of data to read
        /// \endcond
        /// \cond fr
        /// \brief  Lire des donn&eacute;es d'un UserBuffer
        /// \param  aOffsetByte  O&ugrave; d&eacute;buter &agrave;
        ///                      &eacute;crire dans le UserBuffer
        /// \param  aOut         O&ugrave; prendre les donn&eacute;es
        ///                      &agrave; &eacute;crire
        /// \param  aSize_byte   La taille des donn&eacute;es &agrave;
        ///                      &eacute;crire
        /// \endcond
        /// \retval STATUS_OK
        virtual OpenNet::Status Write(unsigned int aOffset_byte, const void * aIn, unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  Delete the instance
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;truire l'instance
        /// \endcond
        virtual void Delete();

    protected:

        UserBuffer();

        virtual ~UserBuffer();

    private:

        UserBuffer(const UserBuffer &);

        const UserBuffer & operator == (const UserBuffer &);

    };

}
