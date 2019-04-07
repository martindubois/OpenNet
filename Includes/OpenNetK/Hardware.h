
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2018-2019 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Hardware.h
/// \brief      OpenNetK::Hardware

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Adapter.h>
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/SpinLock.h>
#include <OpenNetK/Types.h>

namespace OpenNetK
{

    class Packet;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class defines the hardware interface.
    /// \note   This class is part of the Driver Development Kit (DDK).
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe d&eacute;clare l'interface du materiel.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \endcond
    class Hardware
    {

    public:

        /// \cond en
        /// \brief  new operator without allocation
        /// \param  aSize_byte  The size
        /// \param  aAddress    The address
        /// \endcond
        /// \cond fr
        /// \brief  Operateur new sans allocation
        /// \param  aSize_byte  La taille
        /// \param  aAddress    L'adresse
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        void * operator new(size_t aSize_byte, void * aAddress);

        /// \cond en
        /// \brief  Retrieve the common buffer size
        /// \return This methode return the needed common buffer size in
        ///         bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la taille de l'espace m&eacute;moire
        ///         partag&eacute; avec le mat&eacute;riel.
        /// \retval Cette methode retourne la taille n&eacute;cessaire pour
        ///         l'espace m&eacute;moire partage entre le mat&eacute;riel
        ///         et le logiciel.
        /// \endcond
        unsigned int GetCommonBufferSize() const;

        /// \cond en
        /// \brief  Retrieve the configured maximum packet size
        /// \return This methode return the configured maximum packet size in
        ///         bytes.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la taille maximum des paquets configur&eacute;
        /// \retval Cette methode retourne la taille maximum configur&eacute;
        ///         pour les paquets en octes.
        /// \endcond
        unsigned int GetPacketSize() const;

        /// \cond en
        /// \brief  Retrieve the current state
        /// \param  aState  The OpenNet_State instance
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir l'etat courant
        /// \param  aState  L'instance d'OpenNet_State
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Queue
        virtual void GetState(Adapter_State * aState) = 0;

        /// \cond en
        /// \brief  Reset all memory regions
        /// \endcond
        /// \cond fr
        /// \brief  R&eacute;initialiser toutes les regions de m&eacute;moire
        /// \endcond
        /// \note   Level = Thread, Thread = Uninitialisation
        /// \sa     SetMemory
        virtual void ResetMemory();

        /// \cond en
        /// \brief  Connect the adapter
        /// \param  aAdapter  The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Connecter l'Adaptateur
        /// \param  aAdapter  L'Adaptateur
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        virtual void SetAdapter(Adapter * aAdapter);

        /// \cond en
        /// \brief  Set the common buffer
        /// \brief  aCommon_PA  The physical address the hardware uses
        /// \brief  aCommon_CA  The virtual address the software uses
        /// \endcond
        /// \cond fr
        /// \brief  Associe l'espace de memoire contigue
        /// \brief  aCommon_PA  L'adresse physique utilis&eacute;e par le
        ///                     mat&eacute;riel
        /// \brief  aCommon_CA  L'adresse virtuelle utilis&eacute;e par le
        ///                     logiciel
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        virtual void SetCommonBuffer(uint64_t aCommon_PA, void * aCommon_CA);

        /// \cond en
        /// \brief  Set the configuation
        /// \param  aConfig  The configuration
        /// \endcond
        /// \cond fr
        /// \brief  Changer la configuration
        /// \param  aConfig  La configuration
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Users
        virtual void SetConfig(const Adapter_Config & aConfig);

        /// \cond en
        /// \brief  Set a memory region
        /// \param  aIndex      The index
        /// \param  aMemory_MA  The virtual address
        /// \param  aSize_byte  The size
        /// \retval false Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique une r&eacute;gion de m&eacute;moire
        /// \param  aIndex      L'index
        /// \param  aMemory_MA  L'adresse virtuelle
        /// \param  aSize_byte  La taille
        /// \retval false Erreur
        /// \endcond
        /// \retval true  OK
        /// \note   Level = Thread, Thread = Initialisation
        /// \sa     ResetMemory
        virtual bool SetMemory(unsigned int aIndex, void * aMemory_MA, unsigned int aSize_byte);

        /// \cond en
        /// \brief  Enter the D0 state
        /// \endcond
        /// \cond fr
        /// \brief  Entrer dans l'&eacute;tat D0
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        /// \sa     D0_Exit
        virtual void D0_Entry();

        /// \cond en
        /// \brief  Exit the D0 state
        /// \retval false Error
        /// \endcond
        /// \cond fr
        /// \brief  Sortir de l'&eacute;tat D0
        /// \retval false Erreur
        /// \endcond
        /// \retval true  OK
        /// \note   Level = Thread, Thread = Uninitialisation
        /// \sa     D0_Exit
        virtual bool D0_Exit();

        /// \cond en
        /// \brief  Disable the interrupts
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;sactiver les interruptions
        /// \endcond
        /// \sa     Interrupt_Enable
        virtual void Interrupt_Disable();

        /// \cond en
        /// \brief  Enable the interrupts
        /// \endcond
        /// \cond fr
        /// \brief  Activer les interruptions
        /// \endcond
        /// \sa     Interrupt_Disable, Interrupt_Process
        virtual void Interrupt_Enable();

        /// \cond en
        /// \brief  Process an interrupt
        /// \param  aMessageId           The message associated to the
        ///                              interrupt
        /// \param  aNeedMoreProcessing
        /// \retval The adapter did not cause the interrupt.
        /// \retval The adapter caused the interrupt.
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  Traiter une interruption
        /// \param  aMessageId           Le message attach&eacute; a
        ///                              l'interruption
        /// \param  aNeedMoreProcessing
        /// \retval false L'adaptateur n'a pas caus&eacute; l'interruption.
        /// \retval true  L'adaptateur a caus&eacute; l'interruption.
        /// \note   Cette methode fait partie du chemin critique.
        /// \endcond
        /// \sa     Interrupt_Disable, Interrupt_Enable, Interrupt_Process2
        virtual bool Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing);

        /// \cond en
        /// \brief  Process an interrupt at seconde level
        /// \param  aNeedMoreProcessing
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  Traiter une interruption au second niveau
        /// \param  aNeedMoreProcessing
        /// \note   Cette methode fait partie du chemin critique.
        /// \endcond
        /// \sa     Interrupt_Process, Interrupt_Process3
        virtual void Interrupt_Process2(bool * aNeedMoreProcessing);

        /// \cond en
        /// \brief  Process an interrupt at third level
        /// \endcond
        /// \cond fr
        /// \brief  Traiter une interruption au troisi&egrave;me niveau
        /// \endcond
        /// \sa     Interrupt_Process2
        virtual void Interrupt_Process3();

        // TODO  OpenNetK.Hardware
        //       Normal (Optimisation) - Use two lock, one for Rx and one for
        //       Tx

        /// \cond en
        /// \brief  Lock the hardware
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  Verouiller l'acc&egrave;s au mat&eacute;riel
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
        /// \endcond
        /// \sa     Unlock, Unlock_AfterReceive,
        ///         Unlock_AfterReceive_FromThread, Unlock_AfterSend,
        ///         Unlock_AfterSend_FromThread
        void Lock();

        // TODO  OpenNet.Hardware
        //       Normal (Feature) - Ajouter Lock_BeforeSend en passant un
        //       nombre de descripteurs necessaires. Cette fonction echouera
        //       s'il n'y a pas assez de descripteur disponible.

        /// \cond en
        /// \brief  Unlock the hardware
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  D&eacuteverouiller l'acc&egrave;s au mat&eacute;riel
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
        /// \endcond
        /// \sa     Lock
        void Unlock();

        /// \cond en
        /// \brief  Unlock the hardware after programming receive descriptors
        /// \param  aCounter    The counter to increment
        /// \param  aPacketQty  The number of descriptor programmed
        /// \param  aFlags      Value to pass to SpinLock::UnlockFromThread
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;verouiller l'acc&egrave;s au mat&eacute;riel
        ///         apr&egrave;s avoir programm&eacute; des descripteurs de
        ///         r&eacute;ception
        /// \param  aCounter    Le compteur &agrave; incrementer
        /// \param  aPacketQty  Le nombre de descripteurs programm&eacute;s
        /// \param  aFlags      La valeur &agrave; passer &agrave;
        ///                     SpinLock::UnlockFromThread
        /// \endcond
        /// \sa     Lock
        void Unlock_AfterReceive_FromThread(volatile long * aCounter, unsigned int aPacketQty, uint32_t aFlags );

        /// \cond en
        /// \brief  Unlock the hardware after programming transmit descriptors
        /// \param  aCounter    The counter to increment
        /// \param  aPacketQty  The number of descriptor programmed
        /// \param  aFlags      Value to pass to SpinLock::UnlockFromThread
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;verouiller l'acc&egrave;s au mat&eacute;riel
        ///         apr&egrave;s avoir programm&eacute; des descripteurs de
        ///         transmission
        /// \param  aCounter    Le compteur &agrave; incrementer
        /// \param  aPacketQty  Le nombre de descripteurs programm&eacute;s
        /// \param  aFlags      La valeur &agrave; passer &agrave;
        ///                     SpinLock::UnlockFromThread
        /// \endcond
        /// \sa     Lock
        void Unlock_AfterSend_FromThread(volatile long * aCounter, unsigned int aPacketQty, uint32_t aFlags );

        /// \cond en
        /// \brief  Add a buffer to the receiving queue.
        /// \retval false  No available buffer
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter un espace m&eacute;moire &agrave; la queue de
        ///         r&eacute;ception
        /// \retval false  Pas d'espace m&eacute;moire disponible
        /// \endcond
        /// \retval true OK
        virtual bool Packet_Drop() = 0;

        // TODO  OpenNetK.Adapter
        //       Normal (Cleanup) - Pass the aCounter to Lock rather than at
        //       Packet_Receive_NoLock and Unlock_AfterSend and do not pass
        //       aPacketQty to Unlock_AfterSend. Replace it by an internal
        //       counter.

        /// \cond en
        /// \brief  Add the packet to the receiving queue.
        /// \param  aPacket   The Packet
        /// \param  aCounter  The operation counter
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter le paquet &agrave; la queue de r&eacute;ception
        /// \param  aPacket   Le Packet
        /// \param  aCounter  Le compteur d'op&acute;ration
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
        /// \endcond
        /// \sa     Lock, Unlock_AfterReceive
        virtual void Packet_Receive_NoLock(Packet * aPacket, volatile long * aCounter) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param  aPacket_PA  The data
        /// \param  aPacket_XA  The data (C or M)
        /// \param  aSize_byte  The data size
        /// \param  aCounter    The operation counter
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter le paquet &agrave la queue de transmission
        /// \param  aPacket_PA  Les donn&eacute;es
        /// \param  aPacket_XA  Les donn&eacute;es (C or M)
        /// \param  aSize_byte  La taille des donne&eacute;s
        /// \param  aCounter    Le compteur d'op&eacute;ration
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
        /// \endcond
        /// \sa     Lock, Unlock_AfterSend
        virtual void Packet_Send_NoLock(uint64_t aPacket_PA, const void * aPacket_XA, unsigned int aSize_byte, volatile long * aCounter) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param  aPacket       The packet
        /// \param  aSize_byte    The packet size
        /// \param  aRepeatCount  The repeat count
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter le paquet &agrave; la queue de transmission
        /// \param  aPacket       Le paquet
        /// \param  aSize_byte    La taille du paquet
        /// \param  aRepeatCount  Le nombre de r&eacute;p&eacute;tition
        /// \retval false  Erreur
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Users
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
        /// \param  aOut           L'espace m&eacute;moire de sortie
        /// \param  aOutSize_byte  La taille de l'espace m&eacute;moire de
        ///                        sortie
        /// \param  aReset         Remettre les statistiques &agrave;
        ///                        z&eacute;ro apr&egrave;s les avoir obtenus
        /// \return Cette m&eacute;thode retourne la taille des statistiques
        ///         &eacute;crites dans l'espace m&eacute;moire de sortie.
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Users
        virtual unsigned int Statistics_Get(uint32_t * aOut, unsigned int aOutSize_byte, bool aReset);

        /// \cond en
        /// \brief  Reset the statistics
        /// \endcond
        /// \cond fr
        /// \brief  Remettre les statistiques &agrave; z&eacute;ro
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Users
        virtual void Statistics_Reset();

        /// \cond en
        /// \brief  Is Tx enabled?
        /// \endcond
        /// \cond fr
        /// \brief  La transmission est-elle active?
        /// \endcond
        /// \retval false
        /// \retval true
        /// \note   Level = SoftInt or Thread, Thread = Users
        bool Tx_IsEnabled() const;

        /// \cond en
        /// \brief  Disable transmission
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;sactiver la transmission
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Users
        virtual void Tx_Disable();

        /// \cond en
        /// \brief  Enable transmission
        /// \endcond
        /// \cond fr
        /// \brief  Activer la transmission
        /// \endcond
        /// \note   Level = SoftInt or Thread, Thread = Users
        virtual void Tx_Enable();

    // internal:

        void Init(SpinLock * aZone0);

        void GetConfig(Adapter_Config * aConfig);
        void GetInfo  (Adapter_Info   * aInfo  );

        void Tick();

        void Unlock_AfterReceive(volatile long * aCounter, unsigned int aPacketQty);
        void Unlock_AfterSend   (volatile long * aCounter, unsigned int aPacketQty);

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
        /// \brief  Passer les dangereuses barriere de 64 Kio
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
        /// \param  aType             Type of adapter
        /// \param  aPacketSize_byte  The maximum and default packet size
        /// \endcond
        /// \cond fr
        /// \brief  Le constructeur par d&eacute;faut
        /// \param  aType             Type de l'adaptateur
        /// \param  aPacketSize_byte  La taille maximale des paquets et la
        ///                           valeur par defaut pour la configuration
        ///                           de la taille maximale des paquets
        /// \endcond
        /// \note   Thread = Initialisation
        Hardware(OpenNetK::Adapter_Type aType, unsigned int aPacketSize_byte);

        /// \cond en
        /// \brief  Hardware dependent part of the Unlock_AfterReceive
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  La partie de Unlock_AfterReceive qui d&eacute;pend du
        ///         mat&eacute;riel
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
        /// \endcond
        virtual void Unlock_AfterReceive_Internal() = 0;

        /// \cond en
        /// \brief  Hardware dependent part of the Unlock_AfterSend
        /// \note   This method is a part of the critical path.
        /// \endcond
        /// \cond fr
        /// \brief  La partie de Unlock_AfterSend qui d&eacute;pend du
        ///         mat&eacute;riel
        /// \note   Cette m&eacute;thode fait partie du chemin critique.
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
        /// \brief  Le SpinLock utilis&eacute; pour verouille l'acc&egrave;s
        ///         au mat&eacute;riel
        /// \endcond
        SpinLock * mZone0;

    private:

        Hardware(const Hardware &);

        const Hardware & operator = (const Hardware &);

        Adapter * mAdapter;

        bool mTx_Enabled;

    };

    // Public
    /////////////////////////////////////////////////////////////////////////

    inline void Hardware::Lock()
    {
        mZone0->Lock();
    }

    inline void Hardware::Unlock()
    {
        mZone0->Unlock();
    }

    inline bool Hardware::Tx_IsEnabled() const
    {
        return mTx_Enabled;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    inline void Hardware::Tick()
    {
        mAdapter->Tick();
    }

}
