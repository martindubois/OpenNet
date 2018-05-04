
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/System.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNet ===================================================
#include <OpenNet/OpenNet.h>

namespace OpenNet
{

    class Adapter  ;
    class Processor;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the system level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit l'interface au niveau systeme.
    /// \endcond
    class System
    {

    public:

        /// \cond en
        /// \brief   This static methode create an instance of the System
        ///          class.
        /// \return  This static methode return the new instance address.
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode statique cree une instance de la classe
        ///          System.
        /// \return  Cette methode statique retourne l'adresse de la nouvelle
        ///          instance.
        /// \endcond
        static OPEN_NET_PUBLIC System * Create();

        /// \cond en
        /// \brief   This methode delete the instance.
        /// \endcond
        /// \cond fr
        /// \brief   Cette methode detruit l'instance.
        /// \endcond
        virtual void Delete();

        /// \cond en
        /// \return  This methode return the number of adapters.
        /// \endcond
        /// \cond fr
        /// \return  Cette methode retourne le nombre d'adaptateurs.
        /// \endcond
        virtual unsigned int Adapter_GetCount() const = 0;

        /// \cond en
        /// \param   aIndex  The index of the adapter to get
        /// \return  This methode return the adapter.
        /// \endcond
        /// \cond fr
        /// \param   aIndex  L'index de l'adaptateur a retourner
        /// \return  Cette methode retourne l'adaptateurs.
        /// \endcond
        virtual Adapter * Adapter_Get( unsigned int aIndex ) = 0;

        /// \cond en
        /// \return  This methode return the number of processors.
        /// \endcond
        /// \cond fr
        /// \return  Cette methode retourne le nombre de processeurs.
        /// \endcond
        virtual unsigned int Processor_GetCount() const = 0;

        /// \cond en
        /// \param   aIndex  The index of the processor to get
        /// \return  This methode return the processor.
        /// \endcond
        /// \cond fr
        /// \param   aIndex  L'index du processeur a retourner
        /// \return  Cette methode retourne le processeur.
        /// \endcond
        virtual Processor * Processor_Get( unsigned int aIndex ) = 0;

    };

}
