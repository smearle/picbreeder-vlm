/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.gui;

import java.awt.BorderLayout;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import java.awt.GridBagLayout;
import javax.swing.JButton;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.Dimension;
import java.awt.event.KeyEvent;
import java.io.IOException;
import java.io.OutputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;

public class LoginBox extends JDialog {

	private static final long serialVersionUID = 1L;

	private JPanel jContentPane = null;

	private JTextField userField = null;

	private JLabel userLabel = null;

	private JPasswordField passwordField = null;

	private JLabel passwordLabel = null;

	private JButton loginButton = null;

	private JButton cancelButton = null;
	
	private String userName = "";
	
	private String encodedPassword = "";  //  @jve:decl-index=0:
	
	private boolean loginPressed = false;
	
	private boolean registerPressed = false;

	private JButton registerButton = null;
	
	private boolean hasUser = false;
	
	private boolean hasPassword = false;
	
	private char[] password;

	/**
	 * This is the default constructor
	 */
	public LoginBox() {
		super(JOptionPane.getFrameForComponent(client.MainComponentInstance.get()), true);
		initialize();
	}
	
	public void dispose() {
		super.dispose();
		if(password != null)
			java.util.Arrays.fill(password, 0, password.length, '\0');
	}

	/**
	 * This method initializes this
	 * 
	 * @return void
	 */
	private void initialize() {
		this.setSize(240, 102);
		this.setLocation(client.MainComponentInstance.get().getWidth()/2+client.MainComponentInstance.get().getX()-this.getWidth()/2, client.MainComponentInstance.get().getHeight()/2+client.MainComponentInstance.get().getY()-this.getHeight()/2);
		this.setContentPane(getJContentPane());
		this.setTitle("User Information");
		this.setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
		this.requestFocus();
		this.setVisible(true);
	}

	/**
	 * This method initializes jContentPane
	 * 
	 * @return javax.swing.JPanel
	 */
	private JPanel getJContentPane() {
		if (jContentPane == null) {
			GridBagConstraints gridBagConstraints21 = new GridBagConstraints();
			gridBagConstraints21.gridx = 3;
			gridBagConstraints21.fill = GridBagConstraints.BOTH;
			gridBagConstraints21.gridwidth = 1;
			gridBagConstraints21.gridy = 2;
			GridBagConstraints gridBagConstraints1 = new GridBagConstraints();
			gridBagConstraints1.fill = GridBagConstraints.BOTH;
			gridBagConstraints1.gridy = 0;
			gridBagConstraints1.weightx = 1.0;
			gridBagConstraints1.gridheight = 1;
			gridBagConstraints1.gridwidth = 4;
			gridBagConstraints1.anchor = GridBagConstraints.CENTER;
			gridBagConstraints1.gridx = 1;
			GridBagConstraints gridBagConstraints5 = new GridBagConstraints();
			gridBagConstraints5.gridx = 4;
			gridBagConstraints5.fill = GridBagConstraints.HORIZONTAL;
			gridBagConstraints5.gridwidth = 1;
			gridBagConstraints5.gridy = 2;
			GridBagConstraints gridBagConstraints3 = new GridBagConstraints();
			gridBagConstraints3.gridx = 0;
			gridBagConstraints3.fill = GridBagConstraints.HORIZONTAL;
			gridBagConstraints3.gridwidth = 3;
			gridBagConstraints3.gridy = 2;
			GridBagConstraints gridBagConstraints2 = new GridBagConstraints();
			gridBagConstraints2.gridx = 0;
			gridBagConstraints2.anchor = GridBagConstraints.CENTER;
			gridBagConstraints2.fill = GridBagConstraints.HORIZONTAL;
			gridBagConstraints2.gridy = 1;
			passwordLabel = new JLabel();
			passwordLabel.setText("Password:");
			GridBagConstraints gridBagConstraints11 = new GridBagConstraints();
			gridBagConstraints11.fill = GridBagConstraints.BOTH;
			gridBagConstraints11.gridy = 1;
			gridBagConstraints11.weightx = 1.0;
			gridBagConstraints11.anchor = GridBagConstraints.CENTER;
			gridBagConstraints11.gridwidth = 4;
			gridBagConstraints11.gridx = 1;
			GridBagConstraints gridBagConstraints = new GridBagConstraints();
			gridBagConstraints.gridx = 0;
			gridBagConstraints.anchor = GridBagConstraints.CENTER;
			gridBagConstraints.fill = GridBagConstraints.HORIZONTAL;
			gridBagConstraints.gridy = 0;
			userLabel = new JLabel();
			userLabel.setText("Username:");
			userLabel.setDisplayedMnemonic(KeyEvent.VK_UNDEFINED);
			jContentPane = new JPanel();
			jContentPane.setLayout(new GridBagLayout());
			jContentPane.add(userLabel, gridBagConstraints);
			jContentPane.add(getPasswordField(), gridBagConstraints11);
			jContentPane.add(passwordLabel, gridBagConstraints2);
			jContentPane.add(getLoginButton(), gridBagConstraints3);
			jContentPane.add(getCancelButton(), gridBagConstraints5);
			jContentPane.add(getUserField(), gridBagConstraints1);
			jContentPane.add(getRegisterButton(), gridBagConstraints21);
		}
		return jContentPane;
	}

	/**
	 * This method initializes userField	
	 * 	
	 * @return javax.swing.JTextField	
	 */
	private JTextField getUserField() {
		if (userField == null) {
			userField = new JTextField();
			userField.setColumns(20);
			userField.addKeyListener(new java.awt.event.KeyAdapter() {
				public void keyTyped(java.awt.event.KeyEvent e) {
					hasUser = true;
					loginButton.setEnabled(hasUser && hasPassword);
					
					if(e.getKeyChar() == java.awt.event.KeyEvent.VK_ENTER) {
						if(hasPassword)
							loginButton.doClick();
						else
							passwordField.requestFocus();
					}
					
					if(e.getKeyChar() == java.awt.event.KeyEvent.VK_ESCAPE)
						cancelButton.doClick();
				}
			});
		}
		return userField;
	}

	/**
	 * This method initializes passwordField	
	 * 	
	 * @return javax.swing.JPasswordField	
	 */
	private JPasswordField getPasswordField() {
		if (passwordField == null) {
			passwordField = new JPasswordField();
			passwordField.setColumns(20);
			passwordField.addKeyListener(new java.awt.event.KeyAdapter() {
				public void keyTyped(java.awt.event.KeyEvent e) {
					hasPassword = true;
					loginButton.setEnabled(hasUser && hasPassword);
					
					if(e.getKeyChar() == java.awt.event.KeyEvent.VK_ENTER) {
						if(hasUser)
							loginButton.doClick();
						else
							userField.requestFocus();
					}
					
					if(e.getKeyChar() == java.awt.event.KeyEvent.VK_ESCAPE)
						cancelButton.doClick();
				}
			});
		}
		return passwordField;
	}

	/**
	 * This method initializes loginButton	
	 * 	
	 * @return javax.swing.JButton	
	 */
	private JButton getLoginButton() {
		if (loginButton == null) {
			loginButton = new JButton();
			loginButton.setText("Login");
			loginButton.setMnemonic('L');
			loginButton.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent e) {
					userName = userField.getText();
					password = passwordField.getPassword();
					encodedPassword = client.utilities.PasswordEncoder.encode(password);
					loginPressed = true;
					setVisible(false);
				}
			});
			loginButton.setEnabled(false);
		}
		return loginButton;
	}

	/**
	 * This method initializes cancelButton	
	 * 	
	 * @return javax.swing.JButton	
	 */
	private JButton getCancelButton() {
		if (cancelButton == null) {
			cancelButton = new JButton();
			cancelButton.setMnemonic('C');
			cancelButton.setText("Cancel");
			cancelButton.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent e) {
					loginPressed = registerPressed = false;
					setVisible(false);
				}
			});
		}
		return cancelButton;
	}
	
	/**
	 * This method initializes registerButton	
	 * 	
	 * @return javax.swing.JButton	
	 */
	private JButton getRegisterButton() {
		if (registerButton == null) {
			registerButton = new JButton();
			registerButton.setText("Register");
			registerButton.setMnemonic('R');
			registerButton.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent e) {
					registerPressed = true;
					setVisible(false);
				}
			});
		}
		return registerButton;
	}

	
	public String getUserName() {
		return userName;
	}
	
	public String getEncodedPassword() {
		return encodedPassword;
	}
	
	public char[] getRawPassword()
	{
		return password;
	}
	
	public boolean hasLoginInformation() {
		return loginPressed;
	}
	
	public boolean requiresRegistration() {
		return registerPressed;
	}

}  //  @jve:decl-index=0:visual-constraint="190,53"
