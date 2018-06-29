
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Hardware.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Interface.h>
#include <OpenNetK/Types.h>

namespace OpenNetK
{

    class Adapter ;
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
        virtual void GetState(OpenNet_State * aState);

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
        /// \brief  aLogicalAddress           The logical address the
        ///                                   hardware uses
        /// \brief  aVirtualAddress [-K-;RW-] The virtual address the
        ///                                   software uses
        /// \endcond
        /// \cond fr
        /// \brief  Associe l'espace de memoire contigue
        /// \brief  aLogicalAddress           L'adresse logique utilisee par
        //                                    le materiel
        /// \brief  aVirtualAddress [-K-;RW-] L'adresse virtuelle utilisee
        ///                                   par le logiciel
        /// \endcond
        /// \note   Level = Thread, Thread = Initialisation
        virtual void SetCommonBuffer(uint64_t aLogicalAddress, void * aVirtualAddress);

        /// \cond en
        /// \brief  Set the configuation
        /// \param  aConfig [---;R--] La configuration
        /// \endcond
        /// \cond fr
        /// \brief  Connect l'Adaptateur
        /// \param  aConfig [---;R--] La configuration
        /// \endcond
        /// \note   Level = SoftInt, Thread = Queue
        virtual void SetConfig(const OpenNet_Config & aConfig);

        /// \cond en
        /// \brief  Set a memory region
        /// \param  aIndex             The index
        /// \param  aVirtual [-K-;RW-] The virtual address
        /// \param  aSize_byte         The size
        /// \retval false Error
        /// \endcond
        /// \cond fr
        /// \brief  Indique un region de memoire
        /// \param  aIndex             L'index
        /// \param  aVirtual [-K-;RW-] L'adresse virtuelle
        /// \param  aSize_byte         La taille
        /// \retval false Erreur
        /// \endcond
        /// \retval true  OK
        /// \note   Level = Thread, Thread = Initialisation
        virtual bool SetMemory(unsigned int aIndex, void * aVirtual, unsigned int aSize_byte);

        /// \cond en
        /// \brief  Enter the D0 state
        /// \retval false Error
        /// \endcond
        /// \cond fr
        /// \brief  Entrer dans l'etat D0
        /// \retval false Erreur
        /// \endcond
        /// \retval true  OK
        /// \note   Level = Thread, Thread = Initialisation
        virtual bool D0_Entry();

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
        /// \endcond
        /// \cond fr
        /// \brief  Traiter une interruption au second niveau
        /// \endcond
        virtual void Interrupt_Process2();

        /// \cond en
        /// \brief  Add the buffer to the receiving queue.
        /// \param  aLogicalAddress       The data
        /// \param  aPacketInfo [-K-;-W-] The OpenNet_PacketInfo
        /// \param  aCounter    [-K-;RW-] The operation counter
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le buffer a la queue de reception
        /// \param  aLogicalAddress       Les donnees
        /// \param  aPacketInfo [-K-;-W-] Le OpenNet_PacketInfo
        /// \param  aCounter    [-K-;RW-] Le compteur d'operation
        /// \retval false  Erreur
        /// \endcond
        virtual void Packet_Receive(uint64_t aLogicalAddress, OpenNet_PacketInfo * aPacketInfo, volatile long * aCounter) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param  aLogicalAddress    The data
        /// \param  aSize_byte         The data size
        /// \param  aCounter [-K-;RW-] The operation counter
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le paquet a la queue de transmission
        /// \param  aLogicalAddress    Les donnees
        /// \param  aSize_byte         La taille des donnees
        /// \param  aCounter [-K-;RW-] Le compteur d'operation
        /// \endcond
        virtual void Packet_Send(uint64_t aLogicalAddress, unsigned int aSize_byte, volatile long * aCounter) = 0;

        /// \cond en
        /// \brief  Add the packet to the send queue.
        /// \param  aPacket [---;R--] The packet
        /// \param  aSize_byte        The packet size
        /// \endcond
        /// \cond fr
        /// \brief  Ajoute le paquet a la queue de transmission
        /// \param  aPacket [---;R--] Le paquet
        /// \param  aSize_byte        La taille du paquet
        /// \endcond
        /// \note   Thread = Queue
        virtual void Packet_Send(const void * aPacket, unsigned int aSize_byte) = 0;

        virtual void Stats_Get(OpenNet_Stats * aStats, bool aReset);

        virtual void Stats_Reset();

    // internal:

        void Init(SpinLock * aZone0);

        unsigned int GetCommonBufferSize() const;

        void GetConfig(OpenNet_Config * aConfig);
        void GetInfo  (OpenNet_Info   * aInfo  );

    protected:

        /// \cond en
        /// \brief  Skip the dangerous 64 KiB boundaries
        /// \param  aLogical    [---;RW-]
        /// \param  aVirtual    [---;RW-]
        /// \param  aSize_byte
        /// \param  aOutLogical [---;-W-]
        /// \param  aOutVirtual [---;-W-]
        /// \endcond
        /// \cond fr
        /// \brief  Passer les dangereuse barriere de 64 Kio
        /// \param  aLogical    [---;RW-]
        /// \param  aVirtual    [---;RW-]
        /// \param  aSize_byte
        /// \param  aOutLogical [---;-W-]
        /// \param  aOutVirtual [---;-W-]
        /// \endcond
        /// \note   Thread = Initialisation
        static void SkipDangerousBoundary(uint64_t * aLogical, uint8_t ** aVirtual, unsigned int aSize_byte, uint64_t * aOutLogical, uint8_t ** aOutVirtual);

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

        OpenNet_Config mConfig;
        OpenNet_Info   mInfo  ;

        mutable OpenNet_Stats_Hardware         mStats        ;
        mutable OpenNet_Stats_Hardware_NoReset mStats_NoReset;

        // ===== Zone 0 =====================================================
        SpinLock * mZone0;

    private:

        Hardware(const Hardware &);

        const Hardware & operator = (const Hardware &);

        Adapter * mAdapter;

    };

}
