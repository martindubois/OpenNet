
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNetK/Hardware.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Interface.h>

namespace OpenNetK
{

	class Adapter;

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
		void * operator new(size_t aSize_byte, void * aAddress);

		unsigned int GetCommonBufferSize() const;

		/// \cond en
		/// \brief  Retrieve configuration
		/// \param  aConfig [---;-W-] The OpenNet_Config
		/// \endcond
		/// \cond fr
		/// \brief  Retrouver la configuration
		/// \param  aConfig [---;-W-] L'OpenNet_Config
		/// \endcond
		virtual void GetConfig(OpenNet_Config * aConfig);

		/// \cond en
		/// \brief  Retrieve information
		/// \param  aInfo [---;-W-] The OpenNet_Info
		/// \endcond
		/// \cond fr
		/// \brief  Retrouver l'information
		/// \param  aInfo [---;-W-] L'OpenNet_Info
		/// \endcond
		virtual void GetInfo(OpenNet_Info * aInfo);

		unsigned int GetPacketSize() const;

		virtual void GetState(OpenNet_State * aState);

		/// \cond en
		/// \brief  Reset all memory regions
		/// \endcond
		/// \cond fr
		/// \brief  Reset toutes les regions de memoire
		/// \endcond
		/// \note   Thread = Uninitialisation
		virtual void ResetMemory();

		/// \cond en
		/// \brief  Connect the adapter
		/// \param  aAdapter [-K-;RW-] The adapter
		/// \endcond
		/// \cond fr
		/// \brief  Connect l'Adaptateur
		/// \param  aAdapter [-K-;RW-] L'adaptateur
		/// \endcond
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
		/// \note   Thread = Initialisation
		virtual void SetCommonBuffer(uint64_t aLogicalAddress, volatile void * aVirtualAddress);

		/// \cond en
		/// \brief  Set the configuation
		/// \param  aConfig [---;R--] La configuration
		/// \endcond
		/// \cond fr
		/// \brief  Connect l'Adaptateur
		/// \param  aConfig [---;R--] La configuration
		/// \endcond
		virtual void SetConfig(const OpenNet_Config & aConfig);

		/// \cond en
		/// \brief  Set a memory region
		/// \param  aIndex             The index
		/// \param  aVirtual [-K-;RW-] The virtual address
		/// \param  aSize_byte         The size
		/// \endcond
		/// \cond fr
		/// \brief  Indique un region de memoire
		/// \param  aIndex             L'index
		/// \param  aVirtual [-K-;RW-] L'adresse virtuelle
		/// \param  aSize_byte         La taille
		/// \endcond
		/// \note   Thread = Initialisation
		virtual bool SetMemory(unsigned int aIndex, volatile void * aVirtual, unsigned int aSize_byte);

		virtual bool D0_Entry();

		virtual bool D0_Exit();

		virtual void Interrupt_Disable();

		virtual void Interrupt_Enable();

		virtual bool Interrupt_Process(unsigned int aMessageId, bool * aNeedMoreProcessing);

		virtual void Interrupt_Process2();

		/// \cond en
		/// \brief  Add the buffer to the receiving queue.
		/// \param  aBuffer [---;RW-] The buffer
		/// \param  aIndex            The packet index
		/// \retval false  Error
		/// \endcond
		/// \cond fr
		/// \brief  Ajoute le buffer a la queue de reception
		/// \param  aBuffer [---;RW-] Le buffer
		/// \param  aIndex            L'index du paquet
		/// \retval false  Erreur
		/// \endcond
		/// \retval true  OK
		virtual bool Packet_Receive(OpenNet_BufferInfo * aBuffer, unsigned int aIndex) = 0;

		/// \cond en
		/// \brief  Add the packet to the send queue.
		/// \param  aBuffer [---;RW-] The buffer containing the packet
		/// \param  aIndex            The index of the packet into the buffer
		/// \retval false  Error
		/// \endcond
		/// \cond fr
		/// \brief  Ajoute le paquet a la queue de transmission
		/// \param  aBuffer [---;RW-] Le buffer qui contient le paquet
		/// \param  aIndex            L'index du paquet dans le buffer
		/// \retval false  Erreur
		/// \endcond
		/// \retval true  OK
		virtual bool Packet_Send(OpenNet_BufferInfo * aBuffer, unsigned int aIndex) = 0;

		/// \cond en
		/// \brief  Add the packet to the send queue.
		/// \param  aPacket [---;R--] The packet
		/// \param  aSize_byte        The packet size
		/// \retval false  Error
		/// \endcond
		/// \cond fr
		/// \brief  Ajoute le paquet a la queue de transmission
		/// \param  aPacket [---;R--] Le paquet
		/// \param  aSize_byte        La taille du paquet
		/// \retval false  Erreur
		/// \endcond
		/// \retval true  OK
		/// \note   Thread = Queue
		virtual bool Packet_Send(const void * aPacket, unsigned int aSize_byte) = 0;

	protected:

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

		void Skip64KByteBoundary(uint64_t * aLogical, volatile uint8_t ** aVirtual, unsigned int aSize_byte, uint64_t * aOutLogical, volatile uint8_t ** aOutVirtual);

	protected:

		OpenNet_Config mConfig;
		OpenNet_Info   mInfo  ;

	private:

		Hardware(const Hardware &);

		const Hardware & operator = (const Hardware &);

		Adapter * mAdapter;

	};

}
