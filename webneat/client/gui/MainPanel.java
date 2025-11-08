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

import client.evolution.Individual;
import client.evolution.Series;
import javax.swing.JPanel;

class MainPanel extends JPanel {
	MainPanel() {
		super();

		PopulationPanel population = new PopulationPanel();
		this.setLayout(new java.awt.BorderLayout());
		this.add(new EvolutionController(population), java.awt.BorderLayout.NORTH);
		this.add(population, java.awt.BorderLayout.CENTER);
		this.doLayout();
		//this.add(new InformationPanel(), java.awt.BorderLayout.EAST);
	}
}
