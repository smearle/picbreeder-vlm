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

import javax.swing.JComboBox;

public class SubnetComboBox extends JComboBox {
	public SubnetComboBox() {
		addItem("Structure");
		addItem("Color");
		addItem("Both");
		
		setSelectedItem("Both");
		client.evolution.GeneticFactoryInstance.get().setScheme("both");
	}
}
