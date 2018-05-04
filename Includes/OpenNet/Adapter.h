
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Adapter.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNetK ==================================================
#include <OpenNetK/Interface.h>

namespace OpenNet
{

    class Processor;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the adapter level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit l'interface au niveau de l'adaptateur.
    /// \endcond
    class Adapter
    {

    public:

        /// \cond en
        /// \brief  This methode return the information about the adapter.
        /// \param  aOut [---;-W-] The methode return the information here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne les informations au sujet de
        ///        l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les informations ici.
        /// \endcond
        void GetInfo(OpenNet_AdapterInfo * aOut) = 0;

        /// \cond en
        /// \brief  This methode return the statistics of the adapter.
        /// \param  aOut [---;-W-] The methode return the statistics here.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode retourne les statistiques de l'adaptateur.
        /// \param aOut [---;-W-] La medhode retourne les statistiques ici.
        /// \endcond
        void GetStats(OpenNet_AdapterStats * aOut) = 0;

        /// \cond en
        /// \brief  This methode set the input filter.
        /// \param  aOpenCL [---;-W-] The Open CL program.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode affecte le filtre d'entre.
        /// \param aOut [---;-W-] Le programme Open CL
        /// \endcond
        void SetInputFilter(const char * aOpenCL, unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  This methode allocate buffers.
        /// \param  aCount The number of buffer to allocate
        /// \endcond
        /// \cond fr
        /// \brief Cette methode alloue des espace memoire.
        /// \param aCount Le nombre d'espace memoire a allouer
        /// \endcond
        void Buffer_Allocate(unsigned int aCount) = 0;

        /// \cond en
        /// \brief  This methode release buffers.
        /// \param  aCount The number of buffer to release
        /// \endcond
        /// \cond fr
        /// \brief Cette methode relache des espace memoire.
        /// \param aCount Le nombre d'espace memoire a relacher
        /// \endcond
        void Buffer_Release(unsigned int aCount) = 0;

        void Packet_Send(void * aData, unsigned int aSize_byte) = 0;

        /// \cond en
        /// \brief  This methode associate a processor to the adapter.
        /// \param  aProcessor The processor to associate
        /// \endcond
        /// \cond fr
        /// \brief Cette methode associe un processeur a l'adaptateur.
        /// \param aProcessor Le processeur a associer
        /// \endcond
        void Processor_Associate(Processor * aProcessor) = 0;

        /// \cond en
        /// \brief  This methode broke the association to a processor.
        /// \endcond
        /// \cond fr
        /// \brief Cette methode brise l'association a un processeur.
        /// \endcond
        void Processor_Release() = 0;

    };

}
