
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright (C) 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Hardware.h
/// \brief      OpenNetK::Hardware

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/Types.h>

namespace OpenNetK
{

    class Adapter ;
    class Packet  ;
    class SpinLock;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class defines the hardware interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe defenit l'interface du materiel.
    /// \endcond
    class Hardware
    {

    public:

        /// \cond en
        /// \brief  new operator without allocation
        /// \param  aSize_byte         The size
        /// \param  aAddress [---;RW-] The address
        /// \endcond
        /// \cond fr
        /// \brief  Operateur new sans allocation
        /// \param  aSize_byte         La taille
        /// \param  aAddress [---;RW-] L'adresse
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        void * operator new(size_t aSize_byte, void * aAddress);

        /// \cond en
        /// \brief  Retrieve the common buffer size
        /// \return This methode return the needed common buffer size in
        ///         bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la taille de l'espace memoire partage avec le
        ///         hardware.
        /// \retval Cette methode retourne la taille necessaire de l'espace
        ///         memoire partage entre le materiel et le logiciel.
        /// \endcond
        unsigned int GetCommonBufferSize() const;

        /// \cond en
        /// \brief  Retrieve the configured maximum packet size
        /// \return This methode return the configured maximum packet size in
        ///         bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la taille maximum des paquets configure
        /// \retval Cette methode retourne la taille maximum configure pour
        ///         les paquets en octes.
        /// \endcond
        unsigned int GetPacketSize() const;

        /// \cond en
        /// \brief  Retrieve the current state
        /// \param  aState [---;-W-] The OpenNet_State instance
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir l'etat courant
        /// \param  aState [---;-W-] L'instance d'OpenNet_State
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Queue
        virtual void GetState(Adapter_State * aState) = 0;

        /// \cond en
        /// \brief  Reset all memory regions
        /// \endcond
        /// \cond fr
        /// \brief  Reset toutes les regions de memoire
        /// \endcond
        /// \note   Level = Thread, Thread = Uninitialisation
        virtual void ResetMemory();

        /// \cond en
        /// \brief  Connect the adapter
        /// \param  aAdapter [-K-;RW-] The adapter
        /// \endcond
        /// \cond fr
        /// \brief  Connect l'Adaptateur
        /// \param  aAdapter [-K-;RW-] L'adaptateur
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        virtual void SetAdapter(Adapter * aAdapter);

        /// \cond en
        /// \brief  Set the common buffer
        /// \brief  aCommon_PA  The logical address the hardware uses
        /// \brief  aCommon_CA  The virtual address the software uses
        /// \endcond
        /// \cond fr
        /// \brief  Associe l'espace de memoire contigue
        /// \brief  aCommon_PA  L'adresse physique utilisee par le materiel
        /// \brief  aCommon_CA  L'adresse virtuelle utilisee par le logiciel
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        virtual void SetCommonBuffer(uint64_t aCommon_PA, void * aCommon_CA);

        /// \cond en
        /// \brief  Set the configuation
        /// \param  aConfig [---;R--] La configuration
        /// \endcond
        /// \cond fr
        /// \brief  Connect l'Adaptateur
        /// \param  aConfig [---;R--] La configuration
        /// \endcond
        /// \note   Level = SoftInt, Thread = Queue
        virtual void SetConfig(const Adapter_Config & aConfig);

        /// \cond en
        /// \brief  Set a memory region
        /// \param  aIndex      The index
        /// \param  aMemory_MA  The virtual address
        /// \param  aSize_byte  The size
        /// \retval false Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique un region de memoire
        /// \param  aIndex      L'index
        /// \param  aMemory_MA  L'adresse virtuelle
        /// \param  aSize_byte  La taille
        /// \retval false Erreur
        /// \endcond
        /// \retval true  OK
        /// \note   Level = Thread, Thread = Initialisation
        virtual bool SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte);

        /// \cond en
        /// \brief  Enter the D0 state
        /// \endcond
        /// \cond fr
        /// \brief  Entrer dans l'etat D0
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        virtual void D0_Entry();

        /// \cond en
        /// \brief  Exit the D0 state
        /// \retval false Error
        /// \endcond
        /// \cond fr
        /// \brief  Sortir de l'etat D0
        /// \retval false Erreur
        /// \endcond
        /// \retval true  OK
        /// \note   Level = Thread, Thread = Uninitialisation
        virtual bool D0_Exit();

        /// \cond en
        /// \brief  Disable the interrupts
        /// \endcond
        /// \cond fr
        /// \brief  Desactiver les interruption
        /// \endcond
        virtual void Interrupt_Disable();

        /// \cond en
        /// \brief  Enable the interrupts
        /// \endcond
        /// \cond fr
        /// \brief  Activer les interruptions
        /// \endcond
        virtual void Interrupt_Enable();

        /// \cond en
        /// \brief  Process an interrupt
        /// \param  aMessageId                    The message associated to
        ////                                      the interrupt
        /// \param  aNeedMoreProcessing [---;-W-]
        /// \retval The adapter did not cause the interrupt.
        /// \retval The adapter caused the interrupt.
        /// \endcond
        /// \cond fr
        /// \brief  traiter une interruption
        /// \param  aMessageId                    Le message attache a
        ///                                       l'interruption
        /// \param  aNeedMoreProcessing [---;-W-]
        /// \retval false L'adaptateur n'a pas cause l'interruption.
        /// \retval true  L'adaptateur a cause l'interruption.
        /// \endcond
        virtual bool Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing);

        /// \cond en
        /// \brief  Process an interrupt at seconde level
        /// \param  aNeedMoreProcessing [---;-W-]
        /// \endcond
        /// \cond fr
        /// \brief  Traiter une interruption au second niveau
        /// \param  aNeedMoreProcessing [---;-W-]
        /// \endcond
        virtual void Interrupt_Process2(bool * aNeedMoreProcessing);

        /// \cond en
        /// \brief  Process an interrupt at third level
        /// \endcond
        /// \cond fr
        /// \brief  Traiter une interruption au troisieme niveau
        /// \endcond
        virtual void Interrupt_Process3();

        // TODO  OpenNetK.Hardware
        //       Use two lock, one for Rx and one for Tx

        /// \cond en
        /// \brief  Lock the hardware
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller l'acces au materiel
        /// \endcond
        void Lock();

        /// \cond en
        /// \brief  Unlock the hardware
        /// \endcond
        /// \cond fr
        /// \brief  Deverouiller l'acces au materiel
        /// \endcond
        void Unlock();

        /// \cond en
        /// \brief  Unlock the hardware after programming receive descriptors
        /// \param  aCounter    The counter to increment
        /// \param  aPacketQty  The number of descriptor programmed
        /// \endcond
        /// \cond fr
        /// \brief  Deverouiller l'acces au materiel apres avoir programmer
        ///         des descripteurs de reception
        /// \param  aCounter    Le compteur a incrementer
        /// \param  aPacketQty  Le nombre de descripteurs programmes
        /// \endcond
        virtual void Unlock_AfterReceive(volatile long * aCounter, unsigned int aPacketQty);

        /// \cond en
        /// \brief  Unlock the hardware after programming transmit descriptors
        /// \param  aCounter    The counter to increment
        /// \param  aPacketQty  The number of descriptor programmed
        /// \endcond
        /// \cond fr
        /// \brief  Deverouiller l'acces au materiel apres avoir programmer
        ///         des descripteurs de transmission
        /// \param  aCounter    Le compteur a incrementer
        /// \param  aPacketQty  Le nombre de descripteurs programmes
        /// \endcond
        void Unlock_AfterSend(volatile long * aCounter, unsigned int aPacketQty);

        /// \cond en
        /// \brief  Add a buffer to the receiving queue.
        /// \retval false  No available buffer
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute un buffer a la queue de reception
        /// \retval false  Pas d'espace memoire disponible
        /// \endcond
        /// \retval true OK
        virtual bool Packet_Drop() = 0;

        // TODO  OpenNetK.Adapter
        //       Pass the aCounter to Lock rather than at
        //       Packet_Receive_NoLock and Unlock_AfterSend and do not pass
        //       aPacketQty to Unlock_AfterSend. Replace it by an internal
        //       counter.

        /// \cond en
        /// \brief  Add the buffer to the receiving queue.
        /// \param  aPacket   The Packet
        /// \param  aCounter  The operation counter
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le buffer a la queue de reception
        /// \param  aPacket   Le Packet
        /// \param  aCounter  Le compteur d'operation
        /// \endcond
        virtual void Packet_Receive_NoLock(Packet * aPacket, volatile long * aCounter) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param  aPacket_PA  The data
        /// \param  aPacket_XA  The data (C or M)
        /// \param  aSize_byte  The data size
        /// \param  aCounter    The operation counter
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le paquet a la queue de transmission
        /// \param  aPacket_PA  Les donnees
        /// \param  aPacket_XA  Les donnees (C or M)
        /// \param  aSize_byte  La taille des donnees
        /// \param  aCounter    Le compteur d'operation
        /// \endcond
        virtual void Packet_Send_NoLock(uint64_t aPacket_PA, const void * aPacket_XA, unsigned int aSize_byte, volatile long * aCounter) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param  aPacket       The packet
        /// \param  aSize_byte    The packet size
        /// \param  aRepeatCount  The repeat count
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le paquet a la queue de transmission
        /// \param  aPacket       Le paquet
        /// \param  aSize_byte    La taille du paquet
        /// \param  aRepeatCount  Le nombre de repetition
        /// \retval false  Erreur
        /// \endcond
        /// \note   Thread = Queue
        /// \retval true  OK
        virtual bool Packet_Send(const void * aPacket, unsigned int aSize_byte, unsigned int aRepeatCount = 1) = 0;

        /// \cond en
        /// \brief  Retrieve statistics
        /// \param  aOut           The output buffer
        /// \param  aOutSize_byte  The output buffer size
        /// \param  aReset         Reset the statitics after getting them
        /// \return This method returns the size of statistics writen into
        ///         the output buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir les statistiques
        /// \param  aOut           L'espace de memoire de sortie
        /// \param  aOutSize_byte  La taille de l'espace memoire de sortie
        /// \param  aReset         Remettre les statistiques a zero apres les
        ///                        avoir obtenus
        /// \return Cette methode retourne la taille des statistiques ecrites
        ///         dans l'espace de memoire de sortie.
        /// \endcond
        /// \note   Thread = Queue
        virtual unsigned int Statistics_Get(uint32_t * aOut, unsigned int aOutSize_byte, bool aReset);

        /// \cond en
        /// \brief  Reset the statistics
        /// \endcond
        /// \cond fr
        /// \brief  Remettre les statistiques a zero
        /// \endcond
        /// \note   Thread = Queue
        virtual void Statistics_Reset();

    // internal:

        void Init(SpinLock * aZone0);

        void GetConfig(Adapter_Config * aConfig);
        void GetInfo  (Adapter_Info   * aInfo  );

        void Tick();

    protected:

        /// \cond en
        /// \brief  Skip the dangerous 64 KiB boundaries
        /// \param  aIn_PA
        /// \param  aIn_XA      (C or M)
        /// \param  aSize_byte
        /// \param  aOut_PA
        /// \param  aOut_XA     (C or M)
        /// \endcond
        /// \cond fr
        /// \brief  Passer les dangereuse barriere de 64 Kio
        /// \param  aIn_PA
        /// \param  aIn_XA      (C ou M)
        /// \param  aSize_byte
        /// \param  aOut_PA
        /// \param  aOut_XA     (C or M)
        /// \endcond
        /// \note   Thread = Initialisation
        static void SkipDangerousBoundary(uint64_t * aIn_PA, uint8_t ** aIn_XA, unsigned int aSize_byte, uint64_t * aOut_PA, uint8_t ** aOut_XA);

        /// \cond en
        /// \brief  The default constructor
        /// \endcond
        /// \cond fr
        /// \brief  Le constructeur par defaut
        /// \endcond
        /// \note   Thread = Initialisation
        Hardware();

        /// \cond en
        /// \brief  Retrieve the Adapter
        /// \return This method returns the Adapter instance address.
        /// \endcond
        /// \cond fr
        /// \brief  Retrouver l'Adapter
        /// \return Cette methode retourne l'adresse de l'instance d'Adapter.
        /// \endcond
        Adapter * GetAdapter();

        /// \cond en
        /// \brief  Hardware dependent part of the Unlock_AfterReceive
        /// \endcond
        /// \cond fr
        /// \brief  La partie de Unlock_AfterReceive qui depend du materiel
        /// \endcond
        virtual void Unlock_AfterReceive_Internal() = 0;

        /// \cond en
        /// \brief  Hardware dependent part of the Unlock_AfterSend
        /// \endcond
        /// \cond fr
        /// \brief  La partie de Unlock_AfterSend qui depend du materiel
        /// \endcond
        virtual void Unlock_AfterSend_Internal() = 0;

        /// \cond en
        /// \brief  The adapter configuration
        /// \endcond
        /// \cond fr
        /// \brief  La configuration de l'adapteur
        /// \endcond
        Adapter_Config mConfig;

        /// \cond en
        /// \brief  The information about the adapter
        /// \endcond
        /// \cond fr
        /// \brief  L'information au sujet de l'adapteur
        /// \endcond
        Adapter_Info   mInfo  ;

        /// \cond en
        /// \brief  The adapter configuration
        /// \endcond
        /// \cond fr
        /// \brief  La configuration de l'adapteur
        /// \endcond
        mutable uint32_t mStatistics[64];

        // ===== Zone 0 =====================================================

        /// \cond en
        /// \brief  The SpinLock used to lock the hardware
        /// \endcond
        /// \cond fr
        /// \brief  Le SpinLock utilise pour verouille l'access au materiel
        /// \endcond
        SpinLock * mZone0;

    private:

        Hardware(const Hardware &);

        const Hardware & operator = (const Hardware &);

        Adapter * mAdapter;

    };

}
