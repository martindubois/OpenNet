
// Product  OpenNet

/// \author     KMS - Martin Dubois, P.Eng.
/// \copyright  Copyright &copy; 2018-2020 KMS. All rights reserved.
/// \file       Includes/OpenNetK/Adapter.h
/// \brief      OpenNetK::Adapter (DDK)

// CODE REVIEW  2020-04-14  KMS - Martin Dubois, P.Eng.

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNetK/Adapter_Types.h>
#include <OpenNetK/Constants.h>
#include <OpenNetK/IoCtl.h>
#include <OpenNetK/PacketGenerator_Types.h>
#include <OpenNetK/Types.h>

extern "C"
{
    #include <OpenNetK/OSDep.h>
}

namespace OpenNetK
{

    class Hardware;
    class Packet  ;
    class SpinLock;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class maintains information about an adapter on the
    ///         OpenNet internal network.
    /// \note   This class is part of the Driver Development Kit (DDK).
    /// \note   Kernel class - No constructor, no destructor, no virtual
    ///         method
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe maintien les informations concernant un
    ///         adaptateur sur le r&eacute;seau interne OpenNet.
    /// \note   Cette classe fait partie de l'ensemble de developpement de
    ///         pilotes (DDK).
    /// \note   Classe noyau - Pas de constructeur, pas de destructor, pas
    ///         de m&eacute;thodes virtuelle
    /// \endcond
    class Adapter
    {

    public:

        /// \cond en
        /// \brief  Declaration of the event callback
        /// \endcond
        /// \cond fr
        /// \brief  D&eacute;claration de la fonction de traitement des
        ///         evennements
        /// \endcond
        /// \sa     Event_RegisterCallback
        typedef void(*Event_Callback)(void *);

        /// \cond en
        /// \brief  Retrieve information about an IoCtl code
        /// \param  aCode  The IoCtl code
        /// \param  aInfo  The output buffer
        /// \retval false  Error
        /// \endcond
        /// \cond fr
        /// \brief  Optenir l'information au sujet d'un code IoCtl
        /// \param  aCode  Le code IoCtl
        /// \param  aInfo  L'espace m&eacute;moire de sortie
        /// \retval false  Erreur
        /// \endcond
        /// \retval true   OK
        static bool IoCtl_GetInfo(unsigned int aCode, OpenNetK_IoCtl_Info * aInfo);

        /// \cond en
        /// \brief  Cleanup on file close
        /// \param  aFileObject  The closing file
        /// \endcond
        /// \cond fr
        /// \brief  Nettoyer &agrave; la fermeture d'un fichier
        /// \param  aFileObject  Le fichier en fermeture
        /// \endcond
        void FileCleanup( void * aFileObject );

        /// \cond en
        /// \brief  Connect the Hardware instance
        /// \param  aHardware  The Hardware instance
        /// \endcond
        /// \cond fr
        /// \brief  Connecter l'instance de la classe Hardware
        /// \param  aHardware  L'instance de la classe Hardware
        /// \endcond
        /// \note   Level = Thread, Thread = Init
        void SetHardware(Hardware * aHardware);

        /// \cond en
        /// \brief  Set the OSDep structure
        /// \param  aOSDep  The OSDep structure
        /// \endcond
        /// \cond fr
        /// \brief  Assigner la structure OSDep
        /// \param  aOSDep  La structure OSDep
        /// \endcond
        void SetOSDep( OpenNetK_OSDep * aOSDep );

        /// \cond en
        /// \brief  Does some events pending?
        /// \return This method returns the number of currently pending events.
        /// \endcond
        /// \cond fr
        /// \brief  Est-ce que des &eacute;v&eacute;nements sont en attente?
        /// \return Cette m&eacute;thode retourne le nombre
        ///         d'&eacute;v&eacute;nements en attente.
        /// \endcond
        /// \sa     Event_Callback, Event_RegisterCallback
        unsigned int Event_GetPendingCount() const;

        /// \cond en
        /// \brief  Register an event callback
        /// \param  aCallback  The function to call
        /// \param  aContext   The context to pass to the function
        /// \endcond
        /// \cond fr
        /// \brief  Assigner la structure OSDep
        /// \param  aCallback  La fonction
        /// \param  aContext   Le contexte pass&eacute; &agrave; la fonction
        /// \endcond
        /// \sa     Event_GetPendingCount
        void Event_RegisterCallback(Event_Callback aCallback, void * aContext);

    // Internal

        // TODO  OpenNetK.Adapter
        //       Normal (Cleanup) - Definir la structure BufferInfo dans un
        //       fichier prive. En faire une classe.

        /// \cond en
        /// \brief  DirectGMA or GPUDirect Buffer information. This structur
        ///         is not documented and may change or disapear in futur
        ///         version.
        /// \endcond
        /// \cond fr
        /// \brief  L'information au sujet d'un espace m&eacute;moire.
        ///         Cette structure n'est pas document&eacute;. Elle peut
        ///         changer ou dispara&icirc;tre dans une version future.
        /// \endcond
        typedef struct
        {
            Buffer mBuffer;

            uint8_t              * mBase_XA  ; // C or M
            OpenNet_BufferHeader * mHeader_XA; // C or M
            volatile uint32_t    * mMarker_MA;
            OpenNetK::Packet     * mPackets  ;

            struct
            {
                unsigned mStopRequested : 1;

                unsigned mReserved : 31;
            }
            mFlags;

            uint32_t      mEvents     ;
            uint32_t      mMarkerValue;
            unsigned int  mPacketInfoOffset_byte;
            volatile long mRx_Counter ;
            volatile long mTx_Counter ;

            uint32_t mState;

            uint8_t mReserved1[28];
        }
        BufferInfo;

        void Init(SpinLock * aZone0);

        void Buffer_SendPackets(BufferInfo * aBufferInfo);

        void Disconnect();

        void Interrupt_Process2(bool * aNeedMoreProcessing);
        void Interrupt_Process3();

        int  IoCtl( void * aFileObject, unsigned int aCode, const void * aIn, unsigned int aInSize_byte, void * aOut, unsigned int aOutSize_byte);

        void Tick();

    private:

        enum
        {
            EVENT_QTY = 128,
        };

        typedef struct
        {
            unsigned int mCount;
            unsigned int mPx   ;
            unsigned int mRx   ;
            unsigned int mTx   ;
        }
        BufferCountAndIndex;

        void Buffer_InitHeader_Zone0 (OpenNet_BufferHeader * aHeader_XA, const Buffer & aBuffer, Packet * aPackets);
        bool Buffer_Queue_Zone0      (const Buffer & aBuffer);
        void Buffer_Release_Zone0    ();
        void Buffer_Receive_Zone0    (BufferInfo * aBufferInfo);
        void Buffer_Send_Zone0       (BufferInfo * aBufferInfo);
        void Buffer_WriteMarker_Zone0(BufferInfo * aBufferInfo);

        void Event_Report_Zone0(Event_Type aType, uint32_t aData);

        void Interrupt_Process2_Px_Zone0();
        void Interrupt_Process2_Rx_Zone0();
        void Interrupt_Process2_Tx_Zone0();

        void Stop_Zone0();

        // ===== Buffer_ State ==============================================
        void Buffer_EventPending_Zone0(BufferInfo * aBufferInfo);
        void Buffer_PxRunning_Zone0   (BufferInfo * aBufferInfo);
        void Buffer_RxRunning_Zone0   (BufferInfo * aBufferInfo);
        void Buffer_TxRunning_Zone0   (BufferInfo * aBufferInfo);

        // ===== Buffer_Enter_ State ========================================
        void Buffer_Enter_RxProgramming_Zone0(BufferInfo * aBufferInfo, unsigned int aIndex, const char * aFrom);
        void Buffer_Enter_Stopped_Zone0      (BufferInfo * aBufferInfo, unsigned int aIndex, const char * aFrom);

        // ===== IoCtl ======================================================

        int IoCtl_Config_Get(Adapter_Config * aOut);
        int IoCtl_Info_Get  (Adapter_Info   * aOut) const;

        int IoCtl_Config_Set                (const Adapter_Config         * aIn, Adapter_Config         * aOut);
        int IoCtl_PacketGenerator_Config_Set(const PacketGenerator_Config * aIn, PacketGenerator_Config * aOut);

        int IoCtl_Connect(const void * aIn, void * aOut, void * aFileObject);

        int IoCtl_Event_Wait    (const void * aIn, Event    * aOut, unsigned int aOutSize_byte);
        int IoCtl_Statistics_Get(const void * aIn, uint32_t * aOut, unsigned int aOutSize_byte) const;

        int IoCtl_Config_Reset        ();
        int IoCtl_Event_Wait_Cancel   ();
        int IoCtl_Packet_Drop         ();
        int IoCtl_PacketGenerator_Stop();
        int IoCtl_Statistics_Reset    ();
        int IoCtl_Stop                ();
        int IoCtl_Tx_Disable          ();
        int IoCtl_Tx_Enable           ();

        int IoCtl_License_Set(const void * aIn, void * aOut);

        int IoCtl_Packet_Send_Ex(const void * aIn, unsigned int aInSize_byte);

        int IoCtl_PacketGenerator_Config_Get(PacketGenerator_Config * aOut);
        int IoCtl_State_Get                 (Adapter_State          * aOut);

        int IoCtl_PacketGenerator_Start(void * aFileObject);

        int IoCtl_Start(const Buffer * aIn, unsigned int aInSize_byte);

        Adapter   ** mAdapters ;
        unsigned int mAdapterNo;
        Hardware   * mHardware ;
        unsigned int mSystemId ;

        void * mConnect_FileObject;

        PacketGenerator_Config mPacketGenerator_Config ;
        unsigned int           mPacketGenerator_Counter;
        void                 * mPacketGenerator_FileObject;
        long                   mPacketGenerator_Pending;

        mutable uint32_t      mStatistics[32];

        OpenNetK_OSDep * mOSDep;

        mutable uint64_t mStatistics_Start_us;

        uint32_t mEvaluation_ms;
        bool     mLicenseOk    ;

        Adapter_Info mInfo;

        // ===== Zone 0 =====================================================
        SpinLock * mZone0;

        BufferCountAndIndex mBuffer;
        BufferInfo          mBuffers[OPEN_NET_BUFFER_QTY];

        Event_Callback mEvent_Callback;
        void         * mEvent_Context ;
        unsigned int   mEvent_In      ;
        unsigned int   mEvent_Out     ;
        bool           mEvent_Pending ;
        Event          mEvents[EVENT_QTY];

    };

}
