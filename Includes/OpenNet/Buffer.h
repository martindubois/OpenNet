
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/UserBuffer.h
/// \brief      OpenNet::Buffer (SDK)

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
    /// \brief  The Buffer class
    /// \endcond
    /// \cond fr
    /// \brief  La classe Buffer
    /// \endcond
    class Buffer
    {

    public:

        // CRITICAL PATH  BufferEvent  1- / Buffer event

        /// \cond en
        /// \brief  Retrieve the quantity of packet in the buffer
        /// \return This methode returns the packet count.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nombre de paquets
        /// \return This m&eacute; retourne le nombre de paquets.
        /// \endcond
        virtual unsigned int GetPacketCount() const = 0;

        // CRITICAL PATH  BufferEvent  1- / Packet event

        /// \cond en
        /// \brief  Retrieve the packet destination
        /// \param  aIndex  The packet index
        /// \return This methode returns a bit field indicating where the
        ///         packet has been forwarded.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la destination du paquet
        /// \param  aIndex  L'index du paquet
        /// \return This m&eacute; retourne un champ de bits qui indique
        ///         vers quel adaptateur le paquet &agrave;
        ///         &eacute;t&eacute; envoy&eacute;.
        /// \endcond
        virtual uint32_t GetPacketDestination(unsigned int aIndex) const = 0;

        // CRITICAL PATH  BufferEvent  1 / Packet event

        /// \cond en
        /// \brief  Retrieve the index of the next packet event
        /// \param  aIndex  Start searching at this index
        /// \return This methode returns the index of the packet marked for
        ///         event processing or 0xffffffff if no other packet are
        ///         marked.
        /// \endcond
        /// \cond fr
        /// \brief  Trouver le prochain paquet marqu&eacute; pour le
        ///         traintement d'un &eacute;v%eacute;nement
        /// \param  aIndex  L'index de d&eacute;part pour la recherche
        /// \return This m&eacute; retourne l'index du paquet marqu&eacute ou
        ///         0xffffffff s'il n'y en a pas d'autre
        /// \endcond
        virtual unsigned int GetPacketEvent(unsigned int aIndex) const = 0;

        // CRITICAL PATH  BufferEvent

        /// \cond en
        /// \brief  Retrieve the packet size
        /// \param  aIndex  The packet index
        /// \return This methode returns the packet size in byte.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la taille du paquet
        /// \param  aIndex  L'index du paquet
        /// \return This m&eacute; retourne la taille du paquet en octets.
        /// \endcond
        virtual unsigned int GetPacketSize(unsigned int aIndex) const = 0;

        // CRITICAL PATH  BufferEvent  1 / Buffer event

        /// \cond en
        /// \brief  Clear the event
        /// \endcond
        /// \cond fr
        /// \brief  Marquer l'&eacute;v&eacute;nement comme trait&eacute;
        /// \endcond
        /// \retval STATUS_OK
        virtual OpenNet::Status ClearEvent() = 0;

        /// \cond en
        /// \brief  Display
        /// \param  aOut  The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Afficher
        /// \param  aOut  Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        virtual OpenNet::Status Display(FILE * aOut) const = 0;

        // CRITICAL PATH  BufferEvent  1- / Packet event

        /// \cond en
        /// \brief  Read packet
        /// \param  aIndex         Packet index
        /// \param  aOut           The output buffer
        /// \param  aOutSize_byte  The output buffer size
        /// \endcond
        /// \cond fr
        /// \brief  Lire un paquet
        /// \param  aIndex        L'index du paquet &agrave; lire
        /// \param  aOut          L'espace m&eacute;moire de sortie
        /// \param  aOutSize_byte La taille de l'espace de sortie
        /// \endcond
        /// \retval STATUS_OK
        virtual OpenNet::Status ReadPacket(unsigned int aIndex, void * aOut, unsigned int aOutSize_byte) = 0;

        // CRITICAL PATH  BufferEvent  1+ / Buffer event

        /// \cond en
        /// \brief  Wait until all ReadPacket or ClearEvent operations are
        ///         completed
        /// \endcond
        /// \cond fr
        /// \brief  Attendre que toutes les operation ReadPacket ou
        ///         ClearEvent soient termin&eacute;es
        /// \endcond
        /// \retval STATUS_OK
        virtual OpenNet::Status Wait() = 0;

    protected:

        Buffer();

    private:

        Buffer(const Buffer &);

        const Buffer & operator == (const Buffer &);

    };

}
