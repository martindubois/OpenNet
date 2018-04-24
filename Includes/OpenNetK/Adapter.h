
// Product / Produit  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Adapter.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include "Adapter.h"

// Class / Classe
/////////////////////////////////////////////////////////////////////////////

namespace OpenNetK
{

    // =======
    // Adapter
    // =======

    /// \cond en
    /// \brief  This class maintains information about an adapter on the
    ///         OpenNet internal network.
    /// \note   Kernel class - No constructor, No destructor
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe maintien les information concernant un
    ///         adaptateur sur le reseau interne OpenNet.
    /// \note   Classe noyau - Pas de constructeur, Pas de destructeur
    /// \endcond
    class Adapter
    {

    public:

        /// \cond en
        /// \brief  The driver internal information about a buffer
        /// \endcond
        /// \cond fr
        /// \brief  L'information interne au pilote au sujet d'un buffer
        /// \endcond
        typedef struct
        {
            uint32_t mVersion;

            uint8_t mReserved0[4];

            uint64_t               mData_PA;
            OpenNet_BufferHeader * mHeader ;
            volatile uint32_t    * mMarker ;

            uint32_t mAdapterNo   ;
            uint32_t mSendingCount;

            uint8_t mReserved1[16];

            void  * mInternal;
        }
        BufferInfo;

        /// \cond en
        /// \brief  Connect the adapter to the OpenNet internal network
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Connecte l'adaptateur au reseau interne OpenNet.
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        bool Connect();

        /// \cond en
        /// \brief  Disconnect the adapter from the OpenNet internal network
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Deconnecte l'adaptateur du reseau interne OpenNet.
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        bool Disconnect();

        /// \cond en
        /// \brief  Process an IoCtl request
        /// \param aCode           The IoCtl request code
        /// \param aIn  [---;R--]  The input data
        /// \param aInSize_byte    The input data size
        /// \param aOut [---;-W-]  The output data
        /// \param aOutSize_byte   The maximum output data size
        /// \return  If the value is positif, it indicate the output data
        ///          size in bytes. If the value is negatif, it indicate an
        ///          error.
        /// \endcond
        /// \cond fr
        /// \brief  Traite une commande IoCtl
        /// \param aCode  Le code de la commande IoCtl
        /// \param aIn  [---;R--]  Les donnees d'entree
        /// \param aInSize_byte    La taille des donnees d'entree
        /// \param aOut [---;-W-]  Les donnees de sortie
        /// \param aOutSize_byte   La taille maximal des donnes de sortie
        /// \return  Si la valeur est positive, elle represente la taille des
        ///          donnees de sortie en octets. Si elle est negative, elle
        ///          indique une erreur.
        /// \endcond
        int IoCtl(unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte);

        /// \cond en
        /// \brief  Retrieve the minimal input data size for the specified
        ///         command code.
        /// \param aCode  The IoCtl request code
        /// \return  This methode return the minimum size of input data in
        ///          bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Optient la taille minimum des donnees d'entree pour la
        ///         commande specifiee
        /// \param aCode  Le code de la commande IoCtl
        /// \return  Cette methode retourne la taille minimal des donnees
        ///          d'entree en octets.
        /// \endcond
        unsigned int IoCtl_InSize_GetMin(unsigned int aCode);

        /// \cond en
        /// \brief  Retrieve the minimal output data size for the specified
        ///         command code.
        /// \param aCode  The IoCtl request code
        /// \return  This methode return the minimum size of output data in
        ///          bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Optient la taille minimum des donnees de sortie pour la
        ///         commande specifiee
        /// \param aCode  Le code de la commande IoCtl
        /// \return  Cette methode retourne la taille minimal des donnees
        ///          de sortie en octets.
        /// \endcond
        unsigned int IoCtl_OutSize_GetMin(unsigned int aCode);

        /// \cond en
        /// \brief  Indicate the reception of a packet.
        /// \param aBuffer [---;RW-]  The buffer containing the packet
        /// \param aIndex             The index of the packet into the buffer
        /// \pram  aSize_byte         The size of the packet
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique la reception d'un paquet
        /// \param aBuffer [---;RW-]  Le buffer qui contient le paquet
        /// \param aIndex             L'index du paquet dans le buffer
        /// \param aSize_byte         La taille du paquet
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        bool Packet_Received(OpenNet_BufferInfo * aBuffer, unsigned int aIndex, unsigned int aSize_byte);

        /// \cond en
        /// \brief  Indicate an error that occured while receiving a packet.
        /// \param aBuffer [---;RW-]  The buffer containing the packet
        /// \param aIndex             The index of the packet into the buffer
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique une erreur survenue lors de la reception d'un
        ///         paquet
        /// \param aBuffer [---;RW-]  Le buffer qui contient le paquet
        /// \param aIndex             L'index du paquet dans le buffer
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        bool Packet_ReceiveError(OpenNet_BufferInfo * aBuffer, unsigned int aIndex);

        /// \cond en
        /// \brief  Indicate an error that occured while sending a packet.
        /// \param aBuffer [---;RW-]  The buffer containing the packet
        /// \param aIndex             The index of the packet into the buffer
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique une erreur survenue lors de l'envoi d'un
        ///         paquet
        /// \param aBuffer [---;RW-]  Le buffer qui contient le paquet
        /// \param aIndex             L'index du paquet dans le buffer
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        bool Packet_SendError(OpenNet_BufferInfo * aBuffer, unsigned int aIndex);

        /// \cond en
        /// \brief  Indicate the transmission of a packet.
        /// \param aBuffer [---;RW-]  The buffer containing the packet
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique la transmission d'un paquet
        /// \param aBuffer [---;RW-]  Le buffer qui contient le paquet
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        bool Packet_Sent(OpenNet_BufferInfo * aBuffer);

    // Internal

        bool Buffer_Process(BufferInfo * aInfo);

        Adapter * Next_Get();
        void      Next_Set(Adapter * aAdapter);

        Adapter * Previous_Get();
        void      Previous_Set(Adapter * aAdapter);

    protected:

        /// \cond en
        /// \brief  Add the buffer to the receiving queue.
        /// \param aBuffer [---;RW-]  The buffer
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le buffer a la queue de reception
        /// \param aBuffer [---;RW-]  Le buffer
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        virtual bool Buffer_Receive(OpenNet_BufferInfo * aBuffer) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param aBuffer [---;RW-]  The buffer containing the packet
        /// \param aIndex             The index of the packet into the buffer
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le paquet a la queue de transmission
        /// \param aBuffer [---;RW-]  Le buffer qui contient le paquet
        /// \param aIndex             L'index du paquet dans le buffer
        /// \retval false  Erreur
        /// \endcond
        /// \retval true  OK
        virtual bool Packet_Send(OpenNet_BufferInfo * aBuffer, unsigned int aIndex) = 0;

    private:

        bool IoCtl_AdapterInfo_Get   (OpenNet_AdapterInfo  * aOut, unsigned int aOutSize_byte) const;
        bool IoCtl_AdapterStats_Get  (OpenNet_AdapterStats * aOut, unsigned int aOutSize_byte) const;
        bool IoCtl_AdapterStats_Reset();
        bool IoCtl_Buffer_Queue      (const OpenNet_BufferInfo * aIn, unsigned int aInSize_byte);
        bool IoCtl_Buffer_Retrieve   (OpenNet_BufferInfo * aOut, unsigned int aOutSize_byte);
        bool IoCtl_Connect           (OpenNet_IoCtl_Connect_In * aIn, unsigned int aInSize_byte);
        bool IoCtl_Packet_Send       (const void * aIn, unsigned int aInSize_byte);
        bool IoCtl_Version_Get       (OpenNet_Version * aOut, unsigned int aOutSize_byte);

        OpenNet_AdapterInfo mInfo    ;
        Adapter           * mNext    ;
        Adapter           * mPrevious;
        OpenNet_Stats       mStats   ;

    };

}
