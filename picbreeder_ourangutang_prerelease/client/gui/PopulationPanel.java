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

import javax.swing.*;

import java.awt.*;
import java.io.IOException;

import client.ParameterTableInstance;
import client.evolution.*;

public class PopulationPanel extends JPanel {
	private final int rows, columns;
	private final ImagePanel [] images;
	//public EvolutionController evolutionControler;
	
	public int generationID=0;
	
	public PopulationPanel() {
		super();
		
		rows = ParameterTableInstance.get().getInteger("display", "rows");
		columns = ParameterTableInstance.get().getInteger("display", "columns");
		images = new ImagePanel[rows * columns];
		
		this.setLayout(new GridLayout(rows, columns, 1, 1));
		this.setBorder(BorderFactory.createTitledBorder("Population"));
		
		for(int i = 0; i < images.length; i++) {
			images[i] = new ImagePanel(i);
			this.add(images[i]);
		}
	}
	
	public void setEvolutionController(EvolutionController ec)
	{
		for(int i = 0; i < images.length; i++) {
			images[i].evolutionController = ec;
		}
	}
	
	public void setGeneration(Generation p) {

		
		for(int i = 0; i < images.length; i++)
		{
			images[i].setIndividual(p.getIndividual(i));

			images[i].generationID =	generationID;
			//server.getSeries();
			try {
				client.utilities.XML.store(p.getIndividual(i).getGenome(),"genomes/"+EvolutionController.starting_timestamp.getDateTime()+" "+ generationID+"_genome_"+i+".txt");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		generationID++;
	}
	
	public void addMouseListener(java.awt.event.MouseListener listener) {
		super.addMouseListener(listener);
		
		for(ImagePanel p : images)
			p.addMouseListener(listener);
	}
}
