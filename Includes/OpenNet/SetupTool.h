
// Product  OpenNet

/// \author     KMS - Martin Dubois, ing.
/// \copyright  Copyright &copy; 2019 KMS. All rights reserved.
/// \file       Includes/OpenNet/SetupTool.h
/// \brief      OpenNet::SetupTool

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Status.h>

namespace OpenNet
{
    
    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond  en
    /// \brief This class define the setup level interface.
    /// \endcond
    /// \cond  fr
    /// \brief Cette classe definit l'interface au niveau setup.
    /// \endcond
    class SetupTool
    {

    public:

        /// \cond  en
        /// \brief Create an instance of SetupTool
        /// \param aDebug  true to enabled debug output
        /// \endcond
        /// \cond  fr
        /// \brief Cr&eacute;er un instance de SetupTool
        /// \param aDebug  true pour activer les messages de debug
        /// \endcond
        /// \sa    Delete, IsDebugEnabled
        static OPEN_NET_PUBLIC SetupTool * Create(bool aDebug = false);

        /// \cond  en
        /// \brief Create an instance of SetupTool
        /// \endcond
        /// \cond  fr
        /// \brief Cr&eacute;er un instance de SetupTool
        /// \endcond
        /// \sa    Create
        virtual void Delete();

        /// \cond   en
        /// \brief  Retrieve the name of the folder containing binary files.
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nom du r&eacute;pertoire contenant les
        ///         fichiers binaires.
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        virtual const char * GetBinaryFolder() const = 0;

        /// \cond   en
        /// \brief  Retrieve the name of the folder containing header files.
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nom du r&eacute;pertoire contenant les
        ///         fichiers ent&ecirc;te.
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        virtual const char * GetIncludeFolder() const = 0;

        /// \cond   en
        /// \brief  Retrieve the name of the installation folder.
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nom du r&eacute;pertoire d'installation.
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        virtual const char * GetInstallFolder() const = 0;

        /// \cond   en
        /// \brief  Retrieve the name of the folder containing library files.
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nom du r&eacute;pertoire contenant les
        ///         librairies.
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        virtual const char * GetLibraryFolder() const = 0;

        /// \cond   en
        /// \brief  Is the debug trace enabled?
        /// \endcond
        /// \cond   fr
        /// \brief  Les trace de d&eacute;verminage sont-elles
        ///         activ&eacute;es?
        /// \endcond
        /// \retval false
        /// \retval true
        /// \sa     Create
        virtual bool IsDebugEnabled() const = 0;

        /// \cond   en
        /// \brief  Install
        /// \endcond
        /// \cond   fr
        /// \brief  Installer
        /// \endcond
        /// \retval STATUS_OK
        /// \sa     Uninstall
        virtual Status Install() = 0;

        /// \cond   en
        /// \brief  Uninstall
        /// \endcond
        /// \cond   fr
        /// \brief  D&eacute;sinstaller
        /// \endcond
        /// \retval STATUS_OK
        /// \sa     Install
        virtual Status Uninstall() = 0;

        /// \cond   en
        /// \brief  Execute interactif command
        /// \param  aCommand  Index of the command
        /// \endcond
        /// \cond   fr
        /// \brief  Ex&eacute;cuter une commande interactive
        /// \param  aCommand   L'index de la commande
        /// \endcond
        /// \retval STATUS_OK
        /// \sa     Interactif_GetCommand
        virtual Status Interactif_ExecuteCommand(unsigned int aCommand) = 0;

        /// \cond   en
        /// \brief  Retrieve the number of available interactif commands.
        /// \return This method returns the number of commands.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nombre de commandes interactives disponible.
        /// \return Cette m&eacute;thode retourne le nombre de commandes.
        /// \endcond
        /// \sa     Interactif_GetCommand
        virtual unsigned int Interactif_GetCommandCount() = 0;

        /// \cond   en
        /// \brief  Retrieve the text describing an interactif commands.
        /// \param  aCommand  Index of the command
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le texte d&eacute;crivant une commandes
        ///         interactives.
        /// \param  aCommand   L'index de la commande
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        /// \sa     Interactif_ExecuteCommand, Interactif_GetCommandCount
        virtual const char * Interactif_GetCommandText(unsigned int aCommand) = 0;

        /// \cond   en
        /// \brief  Execute a wizard page
        /// \param  aPage    Index of the page
        /// \param  aButton  The button index
        /// \endcond
        /// \cond   fr
        /// \brief  Ex&eacute;cuter une page de l'assistant
        /// \param  aPage    L'index de la page de l'assistant
        /// \param  aButton  L'index du boutton
        /// \endcond
        /// \retval STATUS_OK
        /// \sa     Interractif_GetPageButtonText, Interactif_GetPageText,
        ///         Interactif_GetPageTitle
        virtual Status Wizard_ExecutePage(unsigned int * aPage, unsigned int aButton) = 0;

        /// \cond   en
        /// \brief  Retrieve the number of button on a wizard pages.
        /// \param  aPage  The wizard page index
        /// \return This method returns the number of buttons.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nombre de de bouton dans une page de
        ///         l'assistant.
        /// \param  aPage  L'index de la page de l'assistant
        /// \return Cette m&eacute;thode retourne le nombre de bouttons.
        /// \endcond
        /// \sa     Wizard_GetPageButtonText
        virtual unsigned int Wizard_GetPageButtonCount(unsigned int aPage) = 0;

        /// \cond   en
        /// \brief  Retrieve the text of a button.
        /// \param  aPage    The wizard page index
        /// \param  aButton  The button index
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le texte d'un boutton.
        /// \param  aPage   L'index de la page de l'assistant
        /// \param  aButton  L'index du boutton
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        /// \sa     Wizard_GetPageButtonCount
        virtual const char * Wizard_GetPageButtonText(unsigned int aPage, unsigned int aButton) = 0;

        /// \cond   en
        /// \brief  Retrieve the number of wizard pages.
        /// \return This method returns the number of pages.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le nombre de de page de l'assistant.
        /// \return Cette m&eacute;thode retourne le nombre de pages.
        /// \endcond
        /// \sa     Wizard_GetPageButtonText, Wizard_GetPageText,
        ///         Wizard_GetPageTitle
        virtual unsigned int Wizard_GetPageCount() = 0;

        /// \cond   en
        /// \brief  Retrieve the text of a wizard page.
        /// \param  aPage  The wizard page index
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le texte d'une page de l'assistant.
        /// \param  aPage  L'index de la page de l'assistant
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        /// \sa     Wizard_GetPageButtonText
        virtual const char * Wizard_GetPageText(unsigned int aPage) = 0;

        /// \cond   en
        /// \brief  Retrieve the title of a wizard page.
        /// \param  aPage  The wizard page index
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond   fr
        /// \brief  Obtenir le titre d'une page de l'assistant.
        /// \param  aPage  L'index de la page de l'assistant
        /// \return Cette m&eacute;thode retourne l'adresse d'un espace de
        ///         m&eacute;moire interne.
        /// \endcond
        /// \sa     Wizard_GetPageCount
        virtual const char * Wizard_GetPageTitle(unsigned int aPage) = 0;

    protected:

        SetupTool();

        virtual ~SetupTool();

    private:

        SetupTool(const SetupTool &);

        const SetupTool & operator = (const SetupTool &);

    };

}
